//! Escalation service tests against a stateful stub akhsessd.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::json;

use super::*;
use crate::agent::delegation::native_session::IssuedGrant;
use crate::agent::delegation::types::NativeProjectProfile;
use crate::agent::delegation::{DelegationResult, GrantTools};

const NOON: u64 = 1_790_251_200 + 12 * 3_600;
const REPORT_THREAD: &str = "01a0ab49-9cc4-7012-8785-1eada26fbd46";
const WORK_ITEM: &str = "01a0d569-1cf1-7a2b-8c3d-4e5f60718293";
const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";
const BASE: &str = "89abcdef0123456789abcdef0123456789abcdef";
const REPO: &str = "https://code.aopc.cloud/toasterson/akh-medu.git";

// ---------------------------------------------------------------------------
// Stub akhsessd
// ---------------------------------------------------------------------------

#[derive(Default)]
struct StubState {
    requests: Vec<(String, String, Value)>,
    /// request id → (grant token, session id)
    creates: BTreeMap<String, (String, String)>,
    sessions: BTreeMap<String, Value>,
    runtime_profile: String,
    drop_next_create_answer: bool,
    /// session id → the `BootstrapRepository` intent recorded for it.
    repository: BTreeMap<String, Value>,
    next: u32,
}

struct Stub {
    url: String,
    state: Arc<Mutex<StubState>>,
}

impl Stub {
    fn start(runtime_profile: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(StubState {
            runtime_profile: runtime_profile.into(),
            ..StubState::default()
        }));
        let shared = Arc::clone(&state);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let (method, path, body) = read_request(&mut stream);
                let (status, reply) = handle(&shared, &method, &path, body);
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        Self { url, state }
    }

    fn client(&self) -> NativeSessionClient {
        NativeSessionClient::new(self.url.clone(), "akhsessd-token-long-enough-000")
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StubState> {
        self.state.lock().unwrap()
    }

    /// Requests matching `pattern`: an absolute path ending in `/` matches as
    /// a prefix, any other absolute path exactly, a relative one as a suffix.
    fn matching(&self, method: &str, pattern: &str) -> Vec<Value> {
        let matches = |path: &str| match pattern.strip_prefix('/') {
            Some(_) if pattern.ends_with('/') => path.starts_with(pattern),
            Some(_) if pattern.starts_with("/v1/") => path == pattern,
            _ => path.ends_with(pattern),
        };
        self.lock()
            .requests
            .iter()
            .filter(|(m, p, _)| m == method && matches(p))
            .map(|(_, _, b)| b.clone())
            .collect()
    }

    fn count(&self, method: &str, pattern: &str) -> usize {
        self.matching(method, pattern).len()
    }

    fn bodies(&self, method: &str, pattern: &str) -> Vec<Value> {
        self.matching(method, pattern)
    }

    /// End the first turn of `session` with `status`.
    fn finish(&self, session: &str, status: &str, output: Option<&str>, fault: Option<Value>) {
        let mut state = self.lock();
        let turn = &mut state.sessions.get_mut(session).unwrap()["turns"][0];
        turn["status"] = json!(status);
        turn["output"] = json!(output);
        turn["error"] = fault.unwrap_or(Value::Null);
    }

    /// Record a session as if a create with `request_id` had landed earlier.
    fn seed_create(&self, request_id: &str, grant: &str) -> String {
        let mut state = self.lock();
        state.next += 1;
        let id = format!("sess-seeded-{}", state.next);
        let runtime = state.runtime_profile.clone();
        state.sessions.insert(
            id.clone(),
            json!({"id": id, "turns": [], "runtime": {"profile": runtime, "kind": "codex"}}),
        );
        state
            .creates
            .insert(request_id.into(), (grant.into(), id.clone()));
        id
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> (String, String, Value) {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();
    let mut request = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = stream.read(&mut buf).unwrap_or(0);
        if n == 0 {
            break;
        }
        request.extend_from_slice(&buf[..n]);
        let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&request[..end]).to_string();
        let length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if request.len() >= end + 4 + length {
            break;
        }
    }
    let text = String::from_utf8_lossy(&request).to_string();
    let first = text.lines().next().unwrap_or_default().to_string();
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let body = text
        .split_once("\r\n\r\n")
        .and_then(|(_, b)| serde_json::from_str(b).ok())
        .unwrap_or(Value::Null);
    (method, path, body)
}

fn fault(code: &str) -> String {
    json!({"code": code, "message": code}).to_string()
}

fn handle(state: &Mutex<StubState>, method: &str, path: &str, body: Value) -> (u16, String) {
    let mut s = state.lock().unwrap();
    s.requests
        .push((method.to_string(), path.to_string(), body.clone()));
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    match (method, segments.as_slice()) {
        ("POST", ["v1", "sessions"]) => {
            let request_id = body["request_id"].as_str().unwrap_or_default().to_string();
            let grant = body["grant"].as_str().unwrap_or_default().to_string();
            if let Some((known, id)) = s.creates.get(&request_id).cloned() {
                return if known == grant {
                    (200, s.sessions[&id].to_string())
                } else {
                    (409, fault("conflict"))
                };
            }
            s.next += 1;
            let id = format!("sess-{}", s.next);
            let session = json!({"id": id, "turns": [],
                "runtime": {"profile": s.runtime_profile, "kind": "codex"}});
            s.sessions.insert(id.clone(), session.clone());
            s.creates.insert(request_id, (grant, id));
            if std::mem::take(&mut s.drop_next_create_answer) {
                return (200, "not json".into());
            }
            (200, session.to_string())
        }
        ("GET", ["v1", "session-requests", request_id]) => match s.creates.get(*request_id) {
            Some((_, id)) => (200, s.sessions[id].to_string()),
            None => (404, fault("not_found")),
        },
        ("GET", ["v1", "sessions", id]) => match s.sessions.get(*id) {
            Some(session) => (200, session.to_string()),
            None => (404, fault("not_found")),
        },
        ("POST", ["v1", "sessions", id, "inputs"]) => {
            let request_id = body["request_id"].clone();
            s.next += 1;
            let turn_id = format!("turn-{}", s.next);
            let Some(session) = s.sessions.get_mut(*id) else {
                return (404, fault("not_found"));
            };
            let turns = session["turns"].as_array_mut().unwrap();
            if let Some(turn) = turns.iter().find(|t| t["request_id"] == request_id) {
                return (200, turn.to_string());
            }
            let turn = json!({"id": turn_id, "status": "running", "request_id": request_id});
            turns.push(turn.clone());
            (200, turn.to_string())
        }
        ("POST", ["v1", "sessions", id, "archive"]) => match s.sessions.get_mut(*id) {
            Some(session) => {
                session["closed_at"] = json!(1);
                (200, session.to_string())
            }
            None => (404, fault("not_found")),
        },
        ("POST", ["v1", "sessions", id, "repository"]) => {
            if !s.sessions.contains_key(*id) {
                return (404, fault("not_found"));
            }
            match s.repository.get(*id) {
                Some(existing) if existing == &body => {}
                Some(_) => return (409, fault("conflict")),
                None => {
                    s.repository.insert(id.to_string(), body.clone());
                }
            }
            let receipt = json!({
                "schema_version": 1u16,
                "session_id": id,
                "request": body,
                "principal": "janus",
                "at": 1u64,
            });
            (200, receipt.to_string())
        }
        ("POST", ["v1", "sessions", _, "retire"]) => (200, "{}".into()),
        _ => (404, String::new()),
    }
}

// ---------------------------------------------------------------------------
// Issuer and service fixtures
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Issuer {
    calls: Mutex<Vec<String>>,
    requests: Mutex<Vec<GrantRequest>>,
    next: AtomicU64,
}

impl Issuer {
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl DelegationGrantIssuer for Issuer {
    fn issue(&self, request: &GrantRequest) -> DelegationResult<IssuedGrant> {
        let n = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        self.calls.lock().unwrap().push(format!("issue grant-{n}"));
        self.requests.lock().unwrap().push(request.clone());
        Ok(IssuedGrant {
            id: format!("grant-{n}"),
            akh: "janus".into(),
            token: akh_session_types::GrantToken::new(format!("secret-grant-token-00000000{n}")),
        })
    }
    fn bind_session(&self, grant_id: &str, session_id: &str) -> DelegationResult<()> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("bind {grant_id} {session_id}"));
        Ok(())
    }
    fn revoke(&self, grant_id: &str) -> DelegationResult<()> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("revoke {grant_id}"));
        Ok(())
    }
}

fn skills_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("deploy/roman-mini/skills")
}

fn native_config(runtime_profile: Option<&str>) -> NativeDelegationConfig {
    let mut config = NativeDelegationConfig {
        enabled: true,
        poll_interval_ms: 10,
        ..NativeDelegationConfig::default()
    };
    config.profiles.insert(
        "akh-medu".into(),
        NativeProjectProfile {
            workspace_profile: "janus-worker-akh-medu".into(),
            runtime_profile: runtime_profile.map(str::to_owned),
            repository: None,
        },
    );
    config.escalation.enabled = true;
    config.escalation.escalation_skill = Some(skills_dir().join("astra-escalation"));
    config.escalation.merge_repair_skill = Some(skills_dir().join("mergiraf"));
    config.escalation.report_thread = Some(REPORT_THREAD.into());
    config
}

/// [`native_config`] with `akh-medu`'s profile declaring `repository`
/// (53g PR bootstrap fix).
fn native_config_with_repository(runtime_profile: Option<&str>, repository: &str) -> NativeDelegationConfig {
    let mut config = native_config(runtime_profile);
    config
        .profiles
        .get_mut("akh-medu")
        .expect("akh-medu is always registered by native_config")
        .repository = Some(repository.to_owned());
    config
}

fn lexicon() -> ProviderErrorLexicon {
    ProviderErrorLexicon {
        quota_error_codes: ["usage_limit_exceeded".to_owned()].into(),
        ..ProviderErrorLexicon::default()
    }
}

struct Fixture {
    stub: Stub,
    issuer: Arc<Issuer>,
    service: EscalationService,
    clock: Arc<AtomicU64>,
}

fn fixture_with(stub: Stub, config: &NativeDelegationConfig, ledger: IncidentLedger) -> Fixture {
    let issuer = Arc::new(Issuer::default());
    let clock = Arc::new(AtomicU64::new(NOON));
    let now = Arc::clone(&clock);
    let service = EscalationService::with_client(
        "janus",
        config,
        lexicon(),
        issuer.clone(),
        ledger,
        stub.client(),
    )
    .unwrap()
    .with_clock(Arc::new(move || {
        UnixSecs::from_secs(now.load(Ordering::SeqCst))
    }));
    Fixture {
        stub,
        issuer,
        service,
        clock,
    }
}

fn fixture() -> Fixture {
    fixture_with(
        Stub::start("codex-astra"),
        &native_config(Some("codex-astra")),
        IncidentLedger::in_memory(),
    )
}

fn params(value: Value) -> Map<String, Value> {
    value.as_object().unwrap().clone()
}

fn escalation(key: &str) -> Value {
    json!({
        "incident_key": key,
        "project": "akh-medu",
        "summary": "Closed turn retains ownership",
        "evidence": ["Anima session 01a0d569 completed at 10:00Z but still owns WI"],
        "attempted": ["Read canonical state twice"],
        "requested_action": "Investigate supported reconciliation",
        "work_item_id": WORK_ITEM,
        "pr_number": 412,
        "head_sha": HEAD,
    })
}

fn escalate(fx: &Fixture, key: &str) -> EscalationResult<EscalationReceipt> {
    fx.service
        .escalate(&EscalationRequest::from_params(&params(escalation(key)))?)
}

fn merge_repair() -> Value {
    json!({
        "project": "akh-medu",
        "repository": "toasty/akh-medu",
        "work_item_id": WORK_ITEM,
        "pr_number": 412,
        "head_sha": HEAD,
        "base_sha": BASE,
        "evidence": ["PR #412 reports conflicts in src/lib.rs against main"],
    })
}

fn code_of(error: &EscalationError) -> String {
    error.code_str()
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn escalation_input_is_validated_structurally() {
    assert!(EscalationRequest::from_params(&params(escalation("WI-403-owner"))).is_ok());
    let cases: Vec<(&str, Value, &str)> = vec![
        ("work_item_id", json!("not-a-uuid"), "work_item_id"),
        (
            "work_item_id",
            json!("01a0d5691cf17a2b8c3d4e5f60718293"),
            "work_item_id",
        ),
        ("head_sha", json!("abc123"), "head_sha"),
        ("head_sha", json!(format!("{}zz", &HEAD[..38])), "head_sha"),
        ("pr_number", json!(0), "pr_number"),
        ("pr_number", json!(-3), "pr_number"),
        ("pr_number", json!("12a"), "pr_number"),
        ("anima_session_id", json!(7), "anima_session_id"),
        ("evidence", json!([]), "evidence"),
        ("evidence", json!(vec!["x"; 13]), "evidence"),
        ("evidence", json!(["ok", "  "]), "evidence[1]"),
        ("evidence", json!("not a list"), "evidence"),
        ("attempted", json!(["x".repeat(4_001)]), "attempted[0]"),
        ("summary", json!("s".repeat(161)), "summary"),
        ("requested_action", json!(""), "requested_action"),
        ("surprise", json!("extra"), "surprise"),
    ];
    for (field, value, reported) in cases {
        let mut input = escalation("WI-403-owner");
        input[field] = value.clone();
        let error = EscalationRequest::from_params(&params(input)).unwrap_err();
        match &error {
            EscalationError::InvalidInput { field, .. } => assert_eq!(field, reported, "{value}"),
            other => panic!("{field}={value}: {other:?}"),
        }
        assert_eq!(code_of(&error), "akh::delegation::escalation::input");
        assert!(error.help().is_some());
    }
    // A decimal string PR number is accepted; SHAs are normalised.
    let mut input = escalation("WI-1");
    input["pr_number"] = json!("7");
    input["head_sha"] = json!(HEAD.to_ascii_uppercase());
    let request = EscalationRequest::from_params(&params(input)).unwrap();
    assert_eq!(request.pr_number.unwrap().get(), 7);
    assert_eq!(request.head_sha.unwrap().as_str(), HEAD);
    // Keys: structural rules and the reserved merge-repair namespace.
    let mut input = escalation("x");
    input["incident_key"] = json!("bad key");
    assert!(matches!(
        EscalationRequest::from_params(&params(input)),
        Err(EscalationError::Ledger(LedgerError::InvalidKey { .. }))
    ));
    let mut input = escalation("x");
    input["incident_key"] = json!("merge-repair-abc");
    assert!(matches!(
        EscalationRequest::from_params(&params(input)),
        Err(EscalationError::ReservedKey { .. })
    ));
}

#[test]
fn merge_repair_input_is_validated_and_keyed_by_exact_heads() {
    let request = MergeRepairRequest::from_params(&params(merge_repair())).unwrap();
    let key = request.incident_key();
    assert!(key.is_merge_repair());
    let mut upper = merge_repair();
    upper["head_sha"] = json!(HEAD.to_ascii_uppercase());
    assert_eq!(
        MergeRepairRequest::from_params(&params(upper))
            .unwrap()
            .incident_key(),
        key,
        "case never changes the job identity"
    );
    let mut other = merge_repair();
    other["base_sha"] = json!(HEAD);
    assert_ne!(
        MergeRepairRequest::from_params(&params(other))
            .unwrap()
            .incident_key(),
        key
    );
    for (field, value) in [
        ("pr_number", json!(0)),
        ("base_sha", json!("main")),
        ("work_item_id", json!("WI-403")),
        ("repository", json!("https://user:pass@git.example/a/b")),
        ("repository", json!("../etc")),
        ("repository", json!("no-slash")),
        ("evidence", json!([])),
    ] {
        let mut input = merge_repair();
        input[field] = value.clone();
        let error = MergeRepairRequest::from_params(&params(input)).unwrap_err();
        assert!(
            matches!(&error, EscalationError::InvalidInput { field: f, .. } if f == field),
            "{field}={value}: {error:?}"
        );
    }
    let mut missing = merge_repair();
    missing.as_object_mut().unwrap().remove("head_sha");
    assert!(MergeRepairRequest::from_params(&params(missing)).is_err());
    assert!(
        RepositoryRef::parse("repository", "https://code.aopc.cloud/toasty/akh-medu.git").is_ok()
    );
}

#[test]
fn an_unknown_project_is_refused_before_admission() {
    let fx = fixture();
    let mut input = escalation("WI-1");
    input["project"] = json!("/etc");
    let error = fx
        .service
        .escalate(&EscalationRequest::from_params(&params(input)).unwrap())
        .unwrap_err();
    assert!(matches!(error, EscalationError::UnknownProject { .. }));
    assert!(error.help().unwrap().to_string().contains("akh-medu"));
    assert!(fx.service.ledger().unsettled().is_empty());
    assert_eq!(fx.stub.count("POST", "/v1/sessions"), 0);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

#[test]
fn escalate_creates_one_granted_session_briefed_with_the_skill() {
    let fx = fixture();
    let receipt = escalate(&fx, "WI-403-owner").unwrap();
    assert!(!receipt.duplicate);
    assert_eq!(receipt.state, IncidentState::Running);
    assert_eq!(receipt.workspace_profile, "janus-worker-akh-medu");
    let session = receipt.session_id.clone().unwrap();

    let creates = fx.stub.bodies("POST", "/v1/sessions");
    assert_eq!(creates.len(), 1);
    let key = IncidentKey::parse("WI-403-owner").unwrap();
    assert_eq!(
        creates[0]["request_id"],
        json!(request_id_for("janus", IncidentKind::Escalation, &key))
    );
    assert_eq!(creates[0]["akh"], json!("janus"));
    assert_eq!(
        creates[0]["workspace_profile"],
        json!("janus-worker-akh-medu")
    );
    assert_eq!(creates[0]["grant"], json!("secret-grant-token-000000001"));
    assert!(
        creates[0]
            .get("conversation_key")
            .is_none_or(Value::is_null)
    );

    let grant = fx.issuer.requests.lock().unwrap()[0].clone();
    assert_eq!(grant.principal, "native:akhsessd/janus-worker-akh-medu");
    assert_eq!(grant.service_principal, "sessions");
    assert_eq!(grant.tools, GrantTools::ReadOnlyPlusMemoryWrite);
    assert_eq!(grant.ttl, Duration::from_secs(4 * 3_600 + 300));
    assert_eq!(
        fx.issuer.calls(),
        vec![
            "issue grant-1".to_string(),
            format!("bind grant-1 {session}")
        ]
    );

    let inputs = fx.stub.bodies("POST", "/inputs");
    assert_eq!(inputs.len(), 1);
    let text = inputs[0]["text"].as_str().unwrap();
    let procedures =
        std::fs::read_to_string(skills_dir().join("astra-escalation/procedures.md")).unwrap();
    assert!(
        text.starts_with(procedures.trim_end()),
        "the skill leads the brief"
    );
    let job_start = text.find("```akh-delegation-job\n").unwrap() + 22;
    let job: Value = serde_json::from_str(&text[job_start..text.rfind("```").unwrap()]).unwrap();
    assert_eq!(job["kind"], json!("escalation"));
    assert_eq!(job["incident_key"], json!("WI-403-owner"));
    assert_eq!(job["report_thread"], json!(REPORT_THREAD));
    assert_eq!(job["delegating_akh"], json!("janus"));
    assert_eq!(job["request"]["head_sha"], json!(HEAD));
    assert_eq!(job["delegation_id"], json!(receipt.delegation_id));
    assert!(
        job["skill"]
            .as_str()
            .unwrap()
            .starts_with("astra-escalation@")
    );
}

#[test]
fn a_merge_repair_is_briefed_with_the_mergiraf_skill() {
    let fx = fixture();
    let request = MergeRepairRequest::from_params(&params(merge_repair())).unwrap();
    let receipt = fx.service.dispatch_merge_repair(&request).unwrap();
    assert_eq!(receipt.kind, IncidentKind::MergeRepair);
    assert_eq!(receipt.incident_key, request.incident_key());
    let text = fx.stub.bodies("POST", "/inputs")[0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    let procedures = std::fs::read_to_string(skills_dir().join("mergiraf/procedures.md")).unwrap();
    assert!(text.starts_with(procedures.trim_end()));
    assert!(text.contains(BASE));
    // The same heads again: the same incident, no second worker.
    let again = fx.service.dispatch_merge_repair(&request).unwrap();
    assert!(again.duplicate);
    assert_eq!(fx.stub.count("POST", "/v1/sessions"), 1);
}

#[test]
fn repeats_are_duplicates_and_the_fifth_new_incident_of_a_day_is_refused() {
    let fx = fixture();
    let first = escalate(&fx, "incident-0").unwrap();
    for _ in 0..3 {
        let again = escalate(&fx, "incident-0").unwrap();
        assert!(again.duplicate);
        assert_eq!(again.delegation_id, first.delegation_id);
        assert_eq!(again.session_id, first.session_id);
    }
    for i in 1..4 {
        assert!(!escalate(&fx, &format!("incident-{i}")).unwrap().duplicate);
    }
    let refused = escalate(&fx, "incident-4").unwrap_err();
    assert!(matches!(
        refused,
        EscalationError::Ledger(LedgerError::DailyLimit { limit: 4, .. })
    ));
    assert_eq!(code_of(&refused), "akh::delegation::daily_limit");
    assert_eq!(
        fx.stub.count("POST", "/v1/sessions"),
        4,
        "one session per incident"
    );
    // A new UTC day admits again.
    fx.clock.store(NOON + 86_400, Ordering::SeqCst);
    assert!(!escalate(&fx, "incident-4").unwrap().duplicate);
}

#[test]
fn a_lost_create_answer_is_replayed_in_process_with_the_same_grant() {
    let fx = fixture();
    fx.stub.lock().drop_next_create_answer = true;
    let receipt = escalate(&fx, "WI-lost").unwrap();
    assert_eq!(receipt.state, IncidentState::Running);
    let creates = fx.stub.bodies("POST", "/v1/sessions");
    assert_eq!(creates.len(), 2);
    assert_eq!(
        creates[0], creates[1],
        "identical replay (request id and grant)"
    );
    assert_eq!(fx.issuer.calls()[0], "issue grant-1");
    assert_eq!(
        fx.issuer
            .calls()
            .iter()
            .filter(|c| c.starts_with("issue"))
            .count(),
        1
    );
}

#[test]
fn an_unreachable_akhsessd_leaves_the_incident_admitted_and_pending() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let issuer = Arc::new(Issuer::default());
    let service = EscalationService::with_client(
        "janus",
        &native_config(None),
        lexicon(),
        issuer.clone(),
        IncidentLedger::in_memory(),
        NativeSessionClient::new(url, "akhsessd-token-long-enough-000"),
    )
    .unwrap();
    let request = EscalationRequest::from_params(&params(escalation("WI-offline"))).unwrap();
    let receipt = service.escalate(&request).unwrap();
    assert_eq!(receipt.state, IncidentState::Pending);
    let failure = receipt.dispatch_error.unwrap();
    assert_eq!(failure.code, "akh::delegation::native::transport");
    // Nothing reached akhsessd, so the unused grant was revoked and forgotten.
    let record = service.ledger().get(&request.incident_key).unwrap();
    assert_eq!(record.grant_id, None);
    assert_eq!(record.create_attempts, 1);
    assert!(record.last_error.is_some());
    assert_eq!(issuer.calls(), vec!["issue grant-1", "revoke grant-1"]);
}

#[test]
fn a_session_on_the_wrong_runtime_is_retired_and_the_incident_fails_closed() {
    // akhsessd lacks the worker profile's runtime override: the session would
    // run on the delegating akh's own runtime.
    let fx = fixture_with(
        Stub::start("ollama-janus"),
        &native_config(Some("codex-astra")),
        IncidentLedger::in_memory(),
    );
    let receipt = escalate(&fx, "WI-uncontained").unwrap();
    assert_eq!(receipt.state, IncidentState::Failed);
    assert_eq!(
        receipt.dispatch_error.unwrap().code,
        "akh::delegation::escalation::runtime_mismatch"
    );
    assert_eq!(fx.stub.count("POST", "/inputs"), 0, "no brief was sent");
    assert_eq!(fx.stub.count("POST", "/archive"), 1);
    assert!(fx.issuer.calls().contains(&"revoke grant-1".to_string()));
    let status = fx
        .service
        .status(&IncidentKey::parse("WI-uncontained").unwrap())
        .unwrap();
    assert_eq!(status.next_owner, NextOwner::Operator);
}

// ---------------------------------------------------------------------------
// Repository bootstrap (53g PR bootstrap fix, 2026-09-27 incident)
// ---------------------------------------------------------------------------

#[test]
fn escalate_without_head_sha_is_rejected_when_the_profile_declares_a_repository() {
    let fx = fixture_with(
        Stub::start("codex-astra"),
        &native_config_with_repository(Some("codex-astra"), REPO),
        IncidentLedger::in_memory(),
    );
    let mut without_head = escalation("WI-no-head");
    without_head.as_object_mut().unwrap().remove("head_sha");
    let error = EscalationRequest::from_params(&params(without_head))
        .and_then(|request| fx.service.escalate(&request))
        .unwrap_err();
    assert_eq!(code_of(&error), "akh::delegation::escalation::input");
    assert!(error.to_string().contains("head_sha"), "{error}");
    // The call was refused before admission: no ledger record, no session.
    assert!(
        fx.service
            .ledger()
            .get(&IncidentKey::parse("WI-no-head").unwrap())
            .is_none()
    );
    assert_eq!(fx.stub.count("POST", "/v1/sessions"), 0);
}

#[test]
fn a_project_without_a_repository_still_accepts_escalate_without_head_sha() {
    // The pre-2026-09-27 behavior is unchanged for a project whose profile
    // does not declare a repository.
    let fx = fixture();
    let mut without_head = escalation("WI-no-repo-no-head");
    without_head.as_object_mut().unwrap().remove("head_sha");
    let receipt = fx
        .service
        .escalate(&EscalationRequest::from_params(&params(without_head)).unwrap())
        .unwrap();
    assert_eq!(receipt.state, IncidentState::Running);
    assert_eq!(fx.stub.count("POST", "/repository"), 0);
}

#[test]
fn escalate_with_a_repository_profile_prepares_the_bootstrap_intent_before_the_brief() {
    let fx = fixture_with(
        Stub::start("codex-astra"),
        &native_config_with_repository(Some("codex-astra"), REPO),
        IncidentLedger::in_memory(),
    );
    let receipt = escalate(&fx, "WI-bootstrap").unwrap();
    assert_eq!(receipt.state, IncidentState::Running);
    let prepared = fx.stub.bodies("POST", "/repository");
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0]["checkout"]["repository"], json!(REPO));
    assert_eq!(prepared[0]["checkout"]["revision"], json!(HEAD));
    let requests = fx.stub.lock().requests.clone();
    let repository_at = requests
        .iter()
        .position(|(method, path, _)| method == "POST" && path.ends_with("/repository"))
        .unwrap();
    let inputs_at = requests
        .iter()
        .position(|(method, path, _)| method == "POST" && path.ends_with("/inputs"))
        .unwrap();
    assert!(
        repository_at < inputs_at,
        "the repository intent must reach akhsessd before the brief, or its \
         allocator can stall the session on bootstrap_pending"
    );
}

#[test]
fn a_retried_bootstrap_intent_is_idempotent() {
    let fx = fixture_with(
        Stub::start("codex-astra"),
        &native_config_with_repository(Some("codex-astra"), REPO),
        IncidentLedger::in_memory(),
    );
    let receipt = escalate(&fx, "WI-bootstrap-retry").unwrap();
    let record = fx.service.ledger().get(&receipt.incident_key).unwrap();
    let session_id = record.session_id.clone().unwrap();
    // A retry with the same commit (as a repeated dispatch while the
    // incident was still pending would make) gets back the same receipt
    // instead of a conflict.
    fx.service
        .prepare_repository(&record, &session_id, REPO)
        .unwrap();
    fx.service
        .prepare_repository(&record, &session_id, REPO)
        .unwrap();
    assert_eq!(
        fx.stub.count("POST", "/repository"),
        3,
        "1 from escalate's own dispatch plus 2 direct retries"
    );
}

#[test]
fn a_conflicting_bootstrap_intent_fails_the_incident_closed() {
    let fx = fixture_with(
        Stub::start("codex-astra"),
        &native_config_with_repository(Some("codex-astra"), REPO),
        IncidentLedger::in_memory(),
    );
    // Seed the session id the escalation's create is about to make (the
    // stub's first, deterministic "sess-1") with another repository intent,
    // as if a different principal's create had already reached it.
    fx.stub.lock().repository.insert(
        "sess-1".into(),
        json!({"request_id": "other", "checkout": {"repository": REPO, "revision": BASE}}),
    );
    let receipt = escalate(&fx, "WI-bootstrap-conflict").unwrap();
    assert_eq!(receipt.state, IncidentState::Failed);
    assert_eq!(
        receipt.dispatch_error.unwrap().code,
        "akh::delegation::escalation::bootstrap_conflict"
    );
    assert_eq!(fx.stub.count("POST", "/inputs"), 0, "no brief was sent");
    assert_eq!(fx.stub.count("POST", "/archive"), 1);
    assert!(fx.issuer.calls().contains(&"revoke grant-1".to_string()));
    let status = fx
        .service
        .status(&IncidentKey::parse("WI-bootstrap-conflict").unwrap())
        .unwrap();
    assert_eq!(status.next_owner, NextOwner::Operator);
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

#[test]
fn a_completed_turn_settles_revokes_and_archives() {
    let fx = fixture();
    let receipt = escalate(&fx, "WI-done").unwrap();
    let key = receipt.incident_key.clone();
    let session = receipt.session_id.unwrap();
    let running = fx.service.status(&key).unwrap();
    assert_eq!(running.incident.state, IncidentState::Running);
    assert_eq!(running.next_owner, NextOwner::Worker);
    assert_eq!(fx.stub.count("POST", "/archive"), 0);

    fx.stub.finish(
        &session,
        "completed",
        Some("done\n```akh-result\n{\"status\":\"success\",\"summary\":\"reconciled\"}\n```\n"),
        None,
    );
    let done = fx.service.status(&key).unwrap();
    assert_eq!(done.incident.state, IncidentState::Succeeded);
    assert_eq!(done.incident.job, None, "settled records drop the job");
    assert_eq!(done.next_owner, NextOwner::Akh);
    let worker = done.worker.unwrap();
    assert_eq!(worker.result.unwrap().summary, "reconciled");
    assert!(worker.output_tail.unwrap().starts_with("done"));
    assert!(fx.issuer.calls().contains(&"revoke grant-1".to_string()));
    assert_eq!(fx.stub.count("POST", "/archive"), 1);
    // Reading it again changes nothing.
    let again = fx.service.status(&key).unwrap();
    assert_eq!(again.incident.state, IncidentState::Succeeded);
    assert_eq!(fx.stub.count("POST", "/archive"), 1);
}

#[test]
fn a_provider_caused_failure_is_explained_by_the_taxonomy() {
    let fx = fixture();
    let quota = escalate(&fx, "WI-quota").unwrap();
    fx.stub.finish(
        quota.session_id.as_deref().unwrap(),
        "failed",
        None,
        Some(json!({"code": "usage_limit_exceeded", "message": "limit"})),
    );
    let status = fx.service.status(&quota.incident_key).unwrap();
    assert_eq!(status.incident.state, IncidentState::Failed);
    let failure = status.incident.last_error.clone().unwrap();
    assert_eq!(failure.code, "usage_limit_exceeded");
    assert_eq!(failure.provider_class.as_deref(), Some("quota"));
    assert!(status.provider_blocked);
    assert_eq!(status.next_owner, NextOwner::Operator);

    // A code the lexicon does not list is never guessed from prose.
    let other = escalate(&fx, "WI-other").unwrap();
    fx.stub.finish(
        other.session_id.as_deref().unwrap(),
        "failed",
        None,
        Some(json!({"code": "unknown_outcome", "message": "usage limit reached, HTTP 429"})),
    );
    let status = fx.service.status(&other.incident_key).unwrap();
    assert_eq!(status.incident.state, IncidentState::Failed);
    assert_eq!(status.incident.last_error.unwrap().provider_class, None);
    assert!(!status.provider_blocked);
    assert_eq!(status.next_owner, NextOwner::Akh);

    let cancelled = escalate(&fx, "WI-cancel").unwrap();
    fx.stub.finish(
        cancelled.session_id.as_deref().unwrap(),
        "cancelled",
        None,
        None,
    );
    assert_eq!(
        fx.service
            .status(&cancelled.incident_key)
            .unwrap()
            .incident
            .state,
        IncidentState::Cancelled
    );
}

#[test]
fn an_unknown_incident_is_a_diagnostic() {
    let fx = fixture();
    let error = fx
        .service
        .status(&IncidentKey::parse("never").unwrap())
        .unwrap_err();
    assert_eq!(code_of(&error), "akh::delegation::unknown_incident");
}

// ---------------------------------------------------------------------------
// Restart reconciliation
// ---------------------------------------------------------------------------

fn durable_engine(dir: &std::path::Path) -> Arc<crate::engine::Engine> {
    Arc::new(
        crate::engine::Engine::new(crate::engine::EngineConfig {
            data_dir: Some(dir.to_path_buf()),
            ..crate::engine::EngineConfig::default()
        })
        .unwrap(),
    )
}

/// Admit an incident and record a create attempt, as if the process died
/// right after the create left (the ledger write precedes the send).
fn crashed_after_create(dir: &std::path::Path, key: &str) -> (IncidentRecord, String) {
    let engine = durable_engine(dir);
    let ledger = IncidentLedger::open(&engine).unwrap();
    let key = IncidentKey::parse(key).unwrap();
    let request = EscalationRequest::from_params(&params(escalation(key.as_str()))).unwrap();
    ledger
        .admit(
            NewIncident {
                request_id: request_id_for("janus", IncidentKind::Escalation, &key),
                key: key.clone(),
                kind: IncidentKind::Escalation,
                project: "akh-medu".into(),
                workspace_profile: "janus-worker-akh-medu".into(),
                head_sha: None,
                job: json!({"kind": "escalation", "incident_key": key, "request": request}),
            },
            4,
            UnixSecs::from_secs(NOON),
        )
        .unwrap();
    let record = ledger
        .update(&key, UnixSecs::from_secs(NOON), |r| {
            r.create_attempts = 1;
            r.grant_id = Some("grant-before-restart".into());
        })
        .unwrap();
    engine.persist().unwrap();
    let request_id = record.request_id.clone();
    (record, request_id)
}

#[test]
fn restart_adopts_the_session_an_earlier_create_made() {
    let dir = tempfile::tempdir().unwrap();
    let (record, request_id) = crashed_after_create(dir.path(), "WI-restart");
    let stub = Stub::start("codex-astra");
    let session = stub.seed_create(&request_id, "secret-before-restart-0000000");

    let engine = durable_engine(dir.path());
    let fx = fixture_with(
        stub,
        &native_config(Some("codex-astra")),
        IncidentLedger::open(&engine).unwrap(),
    );
    let report = fx.service.reconcile();
    assert_eq!(
        report,
        ReconcileReport {
            examined: 1,
            dispatched: 1,
            settled: 0,
            still_pending: 0
        }
    );
    assert_eq!(
        fx.stub.count("POST", "/v1/sessions"),
        0,
        "no second create: the session was found by its request id"
    );
    assert_eq!(fx.stub.count("GET", "/v1/session-requests/"), 1);
    assert_eq!(
        fx.issuer.calls(),
        vec![format!("bind grant-before-restart {session}")],
        "no new grant was minted"
    );
    let now = fx.service.ledger().get(&record.incident_key).unwrap();
    assert_eq!(now.state, IncidentState::Running);
    assert_eq!(now.session_id.as_deref(), Some(session.as_str()));
    assert_eq!(now.delegation_id, record.delegation_id);
    assert_eq!(
        fx.stub.count("POST", "/inputs"),
        1,
        "the brief was submitted once"
    );
    // A repeat after the restart is still the same incident.
    assert!(escalate(&fx, "WI-restart").unwrap().duplicate);
    assert_eq!(fx.stub.count("POST", "/v1/sessions"), 0);
}

#[test]
fn restart_creates_the_session_when_no_create_landed() {
    let dir = tempfile::tempdir().unwrap();
    let (record, request_id) = crashed_after_create(dir.path(), "WI-never-sent");
    let engine = durable_engine(dir.path());
    let fx = fixture_with(
        Stub::start("codex-astra"),
        &native_config(Some("codex-astra")),
        IncidentLedger::open(&engine).unwrap(),
    );
    let report = fx.service.reconcile();
    assert_eq!(report.dispatched, 1);
    let creates = fx.stub.bodies("POST", "/v1/sessions");
    assert_eq!(creates.len(), 1);
    assert_eq!(
        creates[0]["request_id"],
        json!(request_id),
        "same request id"
    );
    let calls = fx.issuer.calls();
    assert_eq!(calls[0], "revoke grant-before-restart");
    assert_eq!(calls[1], "issue grant-1");
    let now = fx.service.ledger().get(&record.incident_key).unwrap();
    assert_eq!(now.create_attempts, 2);
    assert_eq!(now.grant_id.as_deref(), Some("grant-1"));
    assert_eq!(now.state, IncidentState::Running);
}

#[test]
fn reconcile_settles_a_turn_that_ended_while_the_akh_was_down() {
    let fx = fixture();
    let receipt = escalate(&fx, "WI-ended").unwrap();
    fx.stub.finish(
        receipt.session_id.as_deref().unwrap(),
        "completed",
        Some("ok"),
        None,
    );
    let report = fx.service.reconcile();
    assert_eq!(report.settled, 1);
    assert!(fx.service.ledger().unsettled().is_empty());
}

// ---------------------------------------------------------------------------
// Configuration and tools
// ---------------------------------------------------------------------------

#[test]
fn the_service_refuses_an_unusable_configuration() {
    let stub = Stub::start("codex-astra");
    let build = |config: &NativeDelegationConfig| {
        EscalationService::with_client(
            "janus",
            config,
            lexicon(),
            Arc::new(Issuer::default()),
            IncidentLedger::in_memory(),
            stub.client(),
        )
    };
    let mut disabled = native_config(None);
    disabled.escalation.enabled = false;
    let mut no_projects = native_config(None);
    no_projects.profiles.clear();
    let mut no_skill = native_config(None);
    no_skill.escalation.merge_repair_skill = None;
    let mut bad_thread = native_config(None);
    bad_thread.escalation.report_thread = Some("status".into());
    for config in [&disabled, &no_projects, &no_skill, &bad_thread] {
        assert!(matches!(build(config), Err(EscalationError::Config { .. })));
    }
    let mut missing_skill = native_config(None);
    missing_skill.escalation.escalation_skill = Some(PathBuf::from("/nonexistent/skill"));
    assert!(matches!(
        build(&missing_skill),
        Err(EscalationError::Skill { .. })
    ));
    assert!(build(&native_config(None)).is_ok());
}

#[test]
fn project_profiles_parse_as_names_or_tables_and_the_legacy_profile_still_parses() {
    let config: crate::agent::delegation::DelegationConfig = toml::from_str(
        "[native]\nenabled = true\nworkspace_profile = \"code\"\nmax_new_per_day = 2\n\
         [native.profiles]\nakh-medu = \"janus-worker-akh-medu\"\n\
         anima = { workspace_profile = \"janus-worker-anima\", runtime_profile = \"codex-astra\", \
         repository = \"https://code.aopc.cloud/toasterson/Anima.git\" }\n\
         [native.escalation]\nenabled = true\nescalation_skill = \"/s/a\"\nmerge_repair_skill = \"/s/m\"\n",
    )
    .unwrap();
    assert_eq!(config.native.workspace_profile, "code");
    assert_eq!(config.native.max_new_per_day, 2);
    assert_eq!(
        config.native.project_profile("akh-medu").unwrap(),
        &NativeProjectProfile {
            workspace_profile: "janus-worker-akh-medu".into(),
            runtime_profile: None,
            repository: None,
        }
    );
    assert_eq!(
        config
            .native
            .project_profile("anima")
            .unwrap()
            .runtime_profile
            .as_deref(),
        Some("codex-astra")
    );
    assert_eq!(
        config
            .native
            .project_profile("anima")
            .unwrap()
            .repository
            .as_deref(),
        Some("https://code.aopc.cloud/toasterson/Anima.git")
    );
    assert!(config.native.escalation.enabled);
    // A config written before PR 7 keeps its single profile and the defaults.
    let legacy: crate::agent::delegation::DelegationConfig =
        toml::from_str("[native]\nenabled = true\nworkspace_profile = \"code\"\n").unwrap();
    assert_eq!(legacy.native.workspace_profile, "code");
    assert!(legacy.native.profiles.is_empty());
    assert_eq!(legacy.native.max_new_per_day, 4);
    assert!(!legacy.native.escalation.enabled);
    // Round trip.
    let text = toml::to_string(&config).unwrap();
    let back: crate::agent::delegation::DelegationConfig = toml::from_str(&text).unwrap();
    assert_eq!(back.native.profiles, config.native.profiles);
}

#[cfg(feature = "core-api")]
#[test]
fn the_tools_are_dangerous_mutating_and_outside_the_legacy_v1_scope() {
    use crate::agent::tool_manifest::{DangerLevel, ToolEffect};
    use crate::core_service::{ToolScope, is_cognitive_read_only};
    let fx = fixture();
    let service = Arc::new(fx.service);
    let tools = crate::agent::tools::escalation_tools(&service);
    let names: Vec<String> = tools.iter().map(|t| t.manifest().name).collect();
    assert_eq!(
        names,
        ["escalate", "dispatch_merge_repair", "escalation_status"]
    );
    for tool in &tools {
        let manifest = tool.manifest();
        assert_eq!(
            manifest.danger.effect,
            ToolEffect::Mutating,
            "{}",
            manifest.name
        );
        assert!(!is_cognitive_read_only(&manifest), "{}", manifest.name);
        assert!(
            !ToolScope::default().permits(&manifest.name),
            "{}",
            manifest.name
        );
        assert!(
            ToolScope::Named {
                tools: [manifest.name.clone()].into()
            }
            .permits(&manifest.name)
        );
    }
    assert_eq!(tools[0].manifest().danger.level, DangerLevel::Dangerous);
    assert_eq!(tools[1].manifest().danger.level, DangerLevel::Dangerous);

    // A bad call is refused with the diagnostic's code and help.
    let engine = crate::engine::Engine::new(crate::engine::EngineConfig::default()).unwrap();
    let mut input = crate::agent::tool::ToolInput::new();
    input.params = params(json!({"incident_key": "x", "project": "akh-medu",
        "summary": "s", "evidence": ["e"], "attempted": ["a"], "requested_action": "r",
        "head_sha": "abc"}));
    let output = tools[0].execute(&engine, input).unwrap();
    assert!(!output.success);
    let refusal: Value = serde_json::from_str(&output.result).unwrap();
    assert_eq!(
        refusal["error"]["code"],
        json!("akh::delegation::escalation::input")
    );
    assert!(
        refusal["error"]["help"]
            .as_str()
            .unwrap()
            .contains("40-hex")
    );

    let mut ok = crate::agent::tool::ToolInput::new();
    ok.params = params(escalation("WI-tool"));
    let output = tools[0].execute(&engine, ok).unwrap();
    assert!(output.success, "{}", output.result);
    let receipt: Value = serde_json::from_str(&output.result).unwrap();
    assert_eq!(receipt["state"], json!("running"));
    let status = tools[2]
        .execute(
            &engine,
            crate::agent::tool::ToolInput::new().with_param("incident_key", "WI-tool"),
        )
        .unwrap();
    assert!(status.success, "{}", status.result);
    let status: Value = serde_json::from_str(&status.result).unwrap();
    assert_eq!(status["incident"]["state"], json!("running"));
    assert_eq!(status["next_owner"], json!("worker"));
}
