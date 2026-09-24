//! Native-session delegation through akhsessd (Phase 53e, ADR 113 D9).
//!
//! The akh delegates a task to a native runtime (Codex, Claude, ...) that
//! akhsessd supervises. Per delegation it:
//!
//! 1. mints a grant through a [`DelegationGrantIssuer`] (the `/v1` core API's
//!    grant authority): read-only cognitive tools plus `memory_write` by
//!    default, presentable only by akhsessd's core principal;
//! 2. creates an akhsessd session carrying the grant (`NewSession.grant`) and
//!    binds the grant to that session;
//! 3. submits the task and polls the session until the turn ends;
//! 4. revokes the grant, whatever happened.
//!
//! The runtime's cognitive tool calls reach the core through akhsessd's broker
//! with the grant attached, so they are checked against the grant's tool list
//! and recorded as the delegate's provenance (`DerivationKind::DelegatedCall`).
//!
//! [`NativeSessionClient`] is a small blocking akhsessd client (ureq). The
//! request bodies are the typed `akh-session-types` contracts; responses are
//! read through tolerant views that only need `id`, `turns[].status` and
//! `turns[].output`, so a newer akhsessd adding fields does not break it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use akh_session_types::{Fault, GrantToken, NewSession, SubmitInput, TurnStatus};
use miette::Diagnostic;
use serde::Deserialize;
use thiserror::Error;

use super::error::DelegationResult;
use super::types::{
    DelegationId, DelegationOutcome, DelegationOutput, DelegationTask, GrantTools,
    NativeDelegationConfig,
};

/// Errors talking to akhsessd for a native-session delegation.
#[derive(Debug, Error, Diagnostic)]
pub enum NativeSessionError {
    #[error("akhsessd token file {path} is not readable")]
    #[diagnostic(
        code(akh::delegation::native::token),
        help(
            "Set `token_file` in [delegation.native] to a file holding this akh's \
             akhsessd bearer token (a principal with the `client` role that may use \
             this akh)."
        )
    )]
    Token {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("akhsessd token is missing or malformed")]
    #[diagnostic(
        code(akh::delegation::native::token_invalid),
        help(
            "The token must be 24+ printable characters without whitespace. Set \
             `token_file` in [delegation.native]."
        )
    )]
    TokenInvalid,

    #[error("akhsessd at {url} is unreachable: {message}")]
    #[diagnostic(
        code(akh::delegation::native::transport),
        help(
            "Check that akhsessd is running and `url` in [delegation.native] points \
             at it. Nothing was delegated if session creation failed."
        )
    )]
    Transport { url: String, message: String },

    #[error("akhsessd rejected {url} ({status}): {code}: {message}")]
    #[diagnostic(
        code(akh::delegation::native::rejected),
        help(
            "Check the akh's akhsessd principal (roles `client`, akh granted), and \
             that `workspace_profile` exists for this akh in akhsessd's config."
        )
    )]
    Rejected {
        url: String,
        status: u16,
        code: String,
        message: String,
    },

    #[error("akhsessd returned an unreadable response from {url}: {message}")]
    #[diagnostic(
        code(akh::delegation::native::decode),
        help("akhsessd and akhomed may be on incompatible versions; deploy them together.")
    )]
    Decode { url: String, message: String },

    #[error("delegated turn {turn_id} in session {session_id} ended {status:?}: {message}")]
    #[diagnostic(
        code(akh::delegation::native::turn_failed),
        help("Inspect the session's events in akhsessd (`GET /v1/sessions/{session_id}/events`).")
    )]
    TurnFailed {
        session_id: String,
        turn_id: String,
        status: TurnStatus,
        message: String,
    },

    #[error("delegated turn in session {session_id} did not finish within {timeout_secs}s")]
    #[diagnostic(
        code(akh::delegation::native::timeout),
        help(
            "Raise `timeout_seconds` in [delegation.native] or split the task. The \
             grant was revoked, so the runtime can no longer reach the akh's tools."
        )
    )]
    Timeout {
        session_id: String,
        timeout_secs: u64,
    },
}

/// A grant request from the delegation manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantRequest {
    /// The delegate the calls are attributed to.
    pub principal: String,
    /// The core principal allowed to present the grant (akhsessd's).
    pub service_principal: String,
    pub tools: GrantTools,
    pub ttl: Duration,
}

/// A minted grant as the delegation manager sees it.
#[derive(Debug, Clone)]
pub struct IssuedGrant {
    /// The grant id (for binding, revocation and provenance).
    pub id: String,
    /// The akh the grant opens.
    pub akh: String,
    /// The secret handed to akhsessd with the session.
    pub token: GrantToken,
}

/// Mints and manages per-delegation grants for one akh. Implemented over the
/// `/v1` core API's grant authority (`core_service::CoreGrantIssuer`).
pub trait DelegationGrantIssuer: Send + Sync {
    fn issue(&self, request: &GrantRequest) -> DelegationResult<IssuedGrant>;
    fn bind_session(&self, grant_id: &str, session_id: &str) -> DelegationResult<()>;
    fn revoke(&self, grant_id: &str) -> DelegationResult<()>;
}

/// The parts of an akhsessd session this client reads.
#[derive(Debug, Clone, Deserialize)]
pub struct SessionView {
    pub id: String,
    #[serde(default)]
    pub turns: Vec<TurnView>,
}

/// The parts of an akhsessd turn this client reads.
#[derive(Debug, Clone, Deserialize)]
pub struct TurnView {
    pub id: String,
    pub status: TurnStatus,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub error: Option<Fault>,
}

/// A small blocking akhsessd client.
#[derive(Clone)]
pub struct NativeSessionClient {
    base_url: String,
    token: String,
    agent: ureq::Agent,
}

impl std::fmt::Debug for NativeSessionClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeSessionClient")
            .field("base_url", &self.base_url)
            .field("token", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl NativeSessionClient {
    /// A client for `base_url` authenticating with `token`.
    pub fn new(base_url: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token: token.into(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(30))
                .build(),
        }
    }

    /// A client from `[delegation.native]`, reading the token file.
    pub fn from_config(config: &NativeDelegationConfig) -> Result<Self, NativeSessionError> {
        let path = config
            .token_file
            .as_deref()
            .ok_or(NativeSessionError::TokenInvalid)?;
        Ok(Self::new(config.url.clone(), read_token(path)?))
    }

    /// `POST /v1/sessions`.
    pub fn create_session(&self, request: &NewSession) -> Result<SessionView, NativeSessionError> {
        self.send("POST", "/v1/sessions", Some(request))
    }

    /// `POST /v1/sessions/{id}/inputs`.
    pub fn submit(
        &self,
        session_id: &str,
        input: &SubmitInput,
    ) -> Result<TurnView, NativeSessionError> {
        self.send(
            "POST",
            &format!("/v1/sessions/{}/inputs", segment(session_id)),
            Some(input),
        )
    }

    /// `GET /v1/sessions/{id}`.
    pub fn session(&self, session_id: &str) -> Result<SessionView, NativeSessionError> {
        self.send::<(), _>(
            "GET",
            &format!("/v1/sessions/{}", segment(session_id)),
            None,
        )
    }

    fn send<B: serde::Serialize, T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        body: Option<&B>,
    ) -> Result<T, NativeSessionError> {
        let url = format!("{}{path}", self.base_url);
        let request = self
            .agent
            .request(method, &url)
            .set("authorization", &format!("Bearer {}", self.token));
        let response = match body {
            Some(body) => {
                let value =
                    serde_json::to_value(body).map_err(|error| NativeSessionError::Decode {
                        url: url.clone(),
                        message: error.to_string(),
                    })?;
                request.send_json(value)
            }
            None => request.call(),
        };
        match response {
            Ok(response) => response
                .into_json::<T>()
                .map_err(|error| NativeSessionError::Decode {
                    url,
                    message: error.to_string(),
                }),
            Err(ureq::Error::Status(status, response)) => {
                let text = response.into_string().unwrap_or_default();
                let fault = serde_json::from_str::<Fault>(&text)
                    .unwrap_or_else(|_| Fault::new("http_error", text));
                Err(NativeSessionError::Rejected {
                    url,
                    status,
                    code: fault.code,
                    message: fault.message,
                })
            }
            Err(error) => Err(NativeSessionError::Transport {
                url,
                message: error.to_string(),
            }),
        }
    }
}

/// Percent-encode one path segment (session ids are UUIDs; this is defence).
fn segment(value: &str) -> String {
    value
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn read_token(path: &Path) -> Result<String, NativeSessionError> {
    let raw = std::fs::read_to_string(path).map_err(|source| NativeSessionError::Token {
        path: path.to_path_buf(),
        source,
    })?;
    let token = raw.trim();
    if token.len() < 24 || token.chars().any(char::is_whitespace) {
        return Err(NativeSessionError::TokenInvalid);
    }
    Ok(token.to_string())
}

/// The `NewSession` request for a delegation: the akh, the profile and the
/// grant, no conversation key (every delegation is its own session).
pub fn new_session_request(
    akh: &str,
    workspace_profile: &str,
    request_id: &str,
    grant: GrantToken,
) -> NewSession {
    NewSession {
        request_id: request_id.into(),
        akh: akh.into(),
        workspace_profile: workspace_profile.into(),
        conversation_key: None,
        grant: Some(grant),
    }
}

/// The delegate identity recorded in provenance for a native session.
pub fn delegate_principal(config: &NativeDelegationConfig) -> String {
    format!("native:{}/{}", config.service, config.workspace_profile)
}

/// Run one native-session delegation to completion (blocking).
pub fn run_native_delegation(
    client: &NativeSessionClient,
    issuer: &dyn DelegationGrantIssuer,
    config: &NativeDelegationConfig,
    task: &DelegationTask,
    id: DelegationId,
) -> DelegationResult<DelegationOutcome> {
    let timeout = Duration::from_secs(config.timeout_seconds.max(1));
    let grant = issuer.issue(&GrantRequest {
        principal: delegate_principal(config),
        service_principal: config.core_principal.clone(),
        tools: config.grant_tools.clone(),
        // The grant outlives the turn by a margin for its final tool calls,
        // and is revoked explicitly when the delegation ends.
        ttl: (timeout + Duration::from_secs(300)).min(Duration::from_secs(24 * 60 * 60)),
    })?;
    let revoke = RevokeOnDrop {
        issuer,
        grant_id: grant.id.clone(),
    };
    let started = Instant::now();
    let request_id = format!("akh-delegation-{}", uuid::Uuid::now_v7());
    let session = client.create_session(&new_session_request(
        &grant.akh,
        &config.workspace_profile,
        &request_id,
        grant.token.clone(),
    ))?;
    issuer.bind_session(&grant.id, &session.id)?;
    let turn = client.submit(
        &session.id,
        &SubmitInput {
            request_id: format!("{request_id}-task"),
            text: build_prompt(task),
            artifacts: vec![],
        },
    )?;
    let poll = Duration::from_millis(config.poll_interval_ms.max(10));
    let finished = loop {
        let current = client.session(&session.id)?;
        if let Some(done) = current
            .turns
            .into_iter()
            .find(|t| t.id == turn.id && t.status.terminal())
        {
            break done;
        }
        if started.elapsed() >= timeout {
            return Err(NativeSessionError::Timeout {
                session_id: session.id,
                timeout_secs: timeout.as_secs(),
            }
            .into());
        }
        std::thread::sleep(poll);
    };
    drop(revoke);
    if finished.status != TurnStatus::Completed {
        return Err(NativeSessionError::TurnFailed {
            session_id: session.id,
            turn_id: finished.id,
            status: finished.status,
            message: finished
                .error
                .map(|fault| fault.to_string())
                .unwrap_or_else(|| "no error reported".into()),
        }
        .into());
    }
    let stdout = finished.output.unwrap_or_default();
    Ok(DelegationOutcome {
        id,
        structured: parse_structured_output(&stdout),
        stdout,
        wall_time: started.elapsed(),
        tokens_consumed: None,
        cost_usd: None,
        exit_code: None,
        goal_id: task.goal_id,
        error: None,
        target: None,
    })
}

/// Revokes the delegation's grant when the delegation ends, on every path.
struct RevokeOnDrop<'a> {
    issuer: &'a dyn DelegationGrantIssuer,
    grant_id: String,
}

impl Drop for RevokeOnDrop<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.issuer.revoke(&self.grant_id) {
            tracing::warn!(
                target: "akh::delegation",
                grant_id = %self.grant_id, %error,
                "failed to revoke delegation grant; it still expires on its TTL"
            );
        }
    }
}

/// The task prompt: the task, KG context, and the structured-result contract.
pub fn build_prompt(task: &DelegationTask) -> String {
    let mut prompt = String::with_capacity(1024);
    prompt.push_str(
        "You are a worker agent delegated a task by an akh, a neuro-symbolic \
         agent. Your session exposes a scoped set of the akh's cognitive tools \
         (knowledge-graph queries, reasoning and memory_write); everything you \
         write through them is recorded as your contribution.\n\n",
    );
    prompt.push_str("## Task\n\n");
    prompt.push_str(&task.description);
    prompt.push_str("\n\n");
    if !task.context_triples.is_empty() {
        prompt.push_str("## Relevant Knowledge (from KG)\n\n");
        for triple in &task.context_triples {
            prompt.push_str(&format!(
                "- {} --{}-> {}\n",
                triple.subject, triple.predicate, triple.object
            ));
        }
        prompt.push('\n');
    }
    prompt.push_str("## Instructions\n\n");
    prompt.push_str("1. Use the akh's cognitive tools to query relevant knowledge.\n");
    prompt.push_str("2. Perform the task.\n");
    prompt.push_str("3. At the end of your response, include a structured result block:\n\n");
    prompt.push_str("```akh-result\n");
    prompt.push_str("{\n");
    prompt.push_str("  \"status\": \"success\" | \"partial\" | \"failed\",\n");
    prompt.push_str("  \"summary\": \"brief description of what was done\",\n");
    prompt.push_str("  \"triples\": [\n");
    prompt.push_str("    {\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}\n");
    prompt.push_str("  ],\n");
    prompt.push_str("  \"files_modified\": [\"path/to/file\"],\n");
    prompt.push_str("  \"follow_up\": \"optional: what should happen next\"\n");
    prompt.push_str("}\n");
    prompt.push_str("```\n");
    prompt
}

/// Parse structured output from the `akh-result` fenced code block.
pub fn parse_structured_output(stdout: &str) -> Option<DelegationOutput> {
    let marker = "```akh-result\n";
    let start = stdout.find(marker)?;
    let rest = stdout.get(start + marker.len()..)?;
    let end = rest.find("```")?;
    serde_json::from_str(rest.get(..end)?).ok()
}

#[cfg(test)]
mod tests {
    use super::super::error::DelegationError;
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[test]
    fn parse_structured_output_valid() {
        let stdout = "preamble\n\n```akh-result\n{\"status\": \"success\", \"summary\": \
                      \"Created 3 test files\", \"triples\": [{\"subject\": \"m\", \
                      \"predicate\": \"has_tests\", \"object\": \"true\"}], \
                      \"files_modified\": [\"src/tests/foo.rs\"]}\n```\n\ntrailing\n";
        let output = parse_structured_output(stdout).unwrap();
        assert_eq!(output.status, "success");
        assert_eq!(output.triples.len(), 1);
        assert_eq!(output.files_modified.len(), 1);
    }

    #[test]
    fn parse_structured_output_missing() {
        assert!(parse_structured_output("Just regular output.").is_none());
    }

    #[test]
    fn build_prompt_includes_context() {
        use crate::skills::LabelTriple;
        let mut task = DelegationTask::new("Write unit tests");
        task.context_triples = vec![LabelTriple {
            subject: "module_a".into(),
            predicate: "has_function".into(),
            object: "calculate".into(),
            confidence: 1.0,
        }];
        let prompt = build_prompt(&task);
        assert!(prompt.contains("Write unit tests"));
        assert!(prompt.contains("module_a --has_function-> calculate"));
        assert!(prompt.contains("akh-result"));
        assert!(!prompt.contains("MCP"), "the /mcp bridge is retired");
    }

    #[test]
    fn the_new_session_request_carries_the_grant_and_no_conversation() {
        let request = new_session_request("minerva", "code", "req-1", GrantToken::new("g"));
        assert_eq!(request.akh, "minerva");
        assert_eq!(request.workspace_profile, "code");
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.conversation_key, None);
        assert_eq!(request.grant, Some(GrantToken::new("g")));
    }

    /// Records what the delegation asked of the grant authority.
    #[derive(Default)]
    struct RecordingIssuer {
        calls: Mutex<Vec<String>>,
        requests: Mutex<Vec<GrantRequest>>,
    }

    impl DelegationGrantIssuer for RecordingIssuer {
        fn issue(&self, request: &GrantRequest) -> DelegationResult<IssuedGrant> {
            self.calls.lock().unwrap().push("issue".into());
            self.requests.lock().unwrap().push(request.clone());
            Ok(IssuedGrant {
                id: "grant-1".into(),
                akh: "minerva".into(),
                token: GrantToken::new("secret-grant-token-0123456789"),
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

    /// A stub akhsessd: answers each request from `script` in order and logs
    /// every raw request.
    fn stub_akhsessd(script: Vec<&'static str>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&log);
        std::thread::spawn(move || {
            for body in script {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                let mut request = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = stream.read(&mut buf).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..n]);
                    let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let headers = String::from_utf8_lossy(&request[..end + 4]).to_string();
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
                seen.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&request).to_string());
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (url, log)
    }

    #[test]
    fn a_native_session_delegation_creates_the_session_with_the_grant_and_revokes_it() {
        let (url, log) = stub_akhsessd(vec![
            r#"{"id":"sess-1","turns":[],"akh":"minerva"}"#,
            r#"{"id":"turn-1","status":"queued"}"#,
            r#"{"id":"sess-1","turns":[{"id":"turn-1","status":"running"}]}"#,
            "{\"id\":\"sess-1\",\"turns\":[{\"id\":\"turn-1\",\"status\":\"completed\",\
             \"output\":\"done\\n```akh-result\\n{\\\"status\\\":\\\"success\\\",\
             \\\"summary\\\":\\\"ok\\\"}\\n```\\n\"}]}",
        ]);
        let client = NativeSessionClient::new(url, "akhsessd-token-long-enough-000");
        let issuer = RecordingIssuer::default();
        let config = NativeDelegationConfig {
            enabled: true,
            workspace_profile: "code".into(),
            poll_interval_ms: 10,
            ..Default::default()
        };
        let task = DelegationTask::new("summarise the incident");
        let outcome =
            run_native_delegation(&client, &issuer, &config, &task, DelegationId(7)).unwrap();
        assert_eq!(outcome.structured.unwrap().summary, "ok");
        assert!(outcome.stdout.starts_with("done"));

        let requests = log.lock().unwrap().clone();
        let create = &requests[0];
        assert!(create.starts_with("POST /v1/sessions "), "{create}");
        assert!(
            create
                .to_ascii_lowercase()
                .contains("authorization: bearer akhsessd-token-long-enough-000"),
            "{create}"
        );
        let body: serde_json::Value =
            serde_json::from_str(create.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        let sent: NewSession = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(sent.akh, "minerva");
        assert_eq!(sent.workspace_profile, "code");
        assert_eq!(
            sent.grant,
            Some(GrantToken::new("secret-grant-token-0123456789"))
        );
        assert!(body.get("conversation_key").is_none_or(|v| v.is_null()));
        assert!(requests[1].starts_with("POST /v1/sessions/sess-1/inputs "));
        assert!(requests[1].contains("summarise the incident"));
        assert!(requests[2].starts_with("GET /v1/sessions/sess-1 "));

        assert_eq!(
            *issuer.calls.lock().unwrap(),
            vec!["issue", "bind grant-1 sess-1", "revoke grant-1"]
        );
        let grant_request = issuer.requests.lock().unwrap()[0].clone();
        assert_eq!(grant_request.principal, "native:akhsessd/code");
        assert_eq!(grant_request.service_principal, "sessions");
        assert_eq!(grant_request.tools, GrantTools::ReadOnlyPlusMemoryWrite);
    }

    #[test]
    fn a_rejected_session_still_revokes_the_grant() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener); // nothing listens: transport error
        let client = NativeSessionClient::new(url, "akhsessd-token-long-enough-000");
        let issuer = RecordingIssuer::default();
        let error = run_native_delegation(
            &client,
            &issuer,
            &NativeDelegationConfig::default(),
            &DelegationTask::new("x"),
            DelegationId(1),
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                DelegationError::NativeSession(NativeSessionError::Transport { .. })
            ),
            "{error:?}"
        );
        assert_eq!(
            *issuer.calls.lock().unwrap(),
            vec!["issue", "revoke grant-1"]
        );
    }
}
