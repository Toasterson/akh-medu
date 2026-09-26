//! Released kagent reads tasks from its REST-backed store. This is a rebuildable
//! projection of central tasks, never an execution owner or an admission retry.
use super::*;
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Config {
    pub controller: Endpoint,
    pub agent_ref: String,
    /// One facade, one dedicated central principal, one kagent user.
    pub user_id: String,
    #[serde(default)]
    pub workspace_profile: Option<String>,
}

pub(super) struct Projection {
    pub config: Config,
    client: reqwest::Client,
    // Serialize fresh reads with writes so concurrent streams cannot overwrite a
    // newer local projection with an older response snapshot. Rebuilt on restart.
    cache: tokio::sync::Mutex<BTreeMap<String, (String, tokio::time::Instant)>>,
}

pub(super) async fn body(mut response: reqwest::Response) -> Result<Value> {
    if !response.status().is_success() {
        return Err(Fault::new(
            "projection_unavailable",
            format!(
                "projection HTTP response was {}",
                response.status().as_u16()
            ),
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| Fault::new("projection_unavailable", "projection response interrupted"))?
    {
        if bytes.len() + chunk.len() > 16 * 1024 * 1024 {
            return Err(Fault::invalid("projection response exceeded byte budget"));
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).map_err(|_| Fault::invalid("invalid projection JSON response"))
}

impl Projection {
    pub fn new(config: Config) -> Result<Self> {
        config.controller.validate()?;
        let parts: Vec<_> = config.agent_ref.split('/').collect();
        if parts.len() != 2
            || parts.iter().any(|p| validate_key(p).is_err())
            || config.user_id.is_empty()
            || config.user_id.len() > 256
            || header::HeaderValue::from_str(&config.user_id).is_err()
        {
            return Err(Fault::invalid(
                "projection requires namespace/agent and one explicit kagent user",
            ));
        }
        Ok(Self {
            client: config.controller.http_client()?,
            config,
            cache: Default::default(),
        })
    }
    fn request(&self, method: reqwest::Method, path: &str) -> Result<reqwest::RequestBuilder> {
        Ok(self
            .client
            .request(method, self.config.controller.route(path))
            .bearer_auth(self.config.controller.token()?)
            .header("X-Agent-Name", &self.config.agent_ref)
            .header("X-User-Id", &self.config.user_id)
            .query(&[("user_id", &self.config.user_id)])
            .timeout(Duration::from_secs(20)))
    }
    async fn ensure_session(&self, context: &str, create: bool) -> Result<()> {
        let response = self
            .request(
                reqwest::Method::GET,
                &format!("/api/sessions/{}", segment(context)),
            )?
            .send()
            .await
            .map_err(|_| {
                Fault::new(
                    "projection_unavailable",
                    "kagent session lookup unavailable",
                )
            })?;
        let session = if response.status() == StatusCode::NOT_FOUND {
            if !create {
                return Err(Fault::missing(
                    "kagent context is not owned by this user/agent",
                ));
            }
            let response = self
                .request(reqwest::Method::POST, "/api/sessions")?
                .json(&json!({"id":context,"agent_ref":self.config.agent_ref,"source":"user"}))
                .send()
                .await
                .map_err(|_| {
                    Fault::new("projection_unavailable", "kagent session save unconfirmed")
                })?;
            body(response).await?["data"].clone()
        } else {
            body(response).await?["data"]["session"].clone()
        };
        if session["id"] != context
            || session["user_id"] != self.config.user_id
            || session["agent_id"]
                != self
                    .config
                    .agent_ref
                    .replace('-', "_")
                    .replace('/', "__NS__")
        {
            return Err(Fault::conflict(
                "kagent session projection has another owner/agent",
            ));
        }
        Ok(())
    }
    fn context_prefix(&self) -> String {
        let identity = serde_json::json!([self.config.agent_ref, self.config.user_id]);
        format!(
            "kagent-{}:",
            akh_session_services::digest(identity.to_string().as_bytes())
        )
    }
    /// kagent UI creates its context before the first prompt. Keep that public
    /// context as an alias of a centrally owned session, durably via conversation_key.
    pub async fn prepare(&self, app: &App, request: &mut Value) -> Result<()> {
        if !matches!(
            request["method"].as_str(),
            Some("message/send" | "message/stream")
        ) {
            return Ok(());
        }
        let Some(context) = request["params"]["message"]["contextId"]
            .as_str()
            .map(str::to_owned)
        else {
            return Ok(());
        };
        validate_key(&context)?;
        if context.len() > 128 {
            return Err(Fault::invalid("kagent context exceeds alias budget"));
        }
        self.ensure_session(&context, false).await?;
        let response = app
            .client
            .get(
                app.config
                    .sessions
                    .route(&format!("/v1/sessions/{}", segment(&context))),
            )
            .bearer_auth(app.config.sessions.token()?)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(|_| {
                Fault::new(
                    "projection_unavailable",
                    "central context lookup unavailable",
                )
            })?;
        let session: SessionRecord = if response.status() == StatusCode::NOT_FOUND {
            let profile = self.config.workspace_profile.as_ref().ok_or_else(|| {
                Fault::new(
                    "unsupported",
                    "configure the workspace profile for kagent-created contexts",
                )
            })?;
            let conversation = format!("{}{context}", self.context_prefix());
            let response = app
                .client
                .post(app.config.sessions.route("/v1/sessions"))
                .bearer_auth(app.config.sessions.token()?)
                .timeout(Duration::from_secs(20))
                .json(&NewSession {
                    request_id: conversation.clone(),
                    akh: app.config.akh.clone(),
                    workspace_profile: profile.clone(),
                    conversation_key: Some(conversation.clone()),
                    grant: None,
                })
                .send()
                .await
                .map_err(|_| {
                    Fault::new(
                        "projection_unavailable",
                        "central context admission unconfirmed",
                    )
                })?;
            let session: SessionRecord = serde_json::from_value(body(response).await?)
                .map_err(|_| Fault::invalid("invalid central context"))?;
            if session.conversation_key.as_deref() != Some(&conversation) {
                return Err(Fault::conflict("central context alias differs"));
            }
            session
        } else {
            serde_json::from_value(body(response).await?)
                .map_err(|_| Fault::invalid("invalid central context"))?
        };
        if session.akh != app.config.akh {
            return Err(Fault::conflict("context belongs to another Akh"));
        }
        request["params"]["message"]["contextId"] = json!(session.id);
        Ok(())
    }
    pub async fn publish(&self, app: &App, id: &str) -> Result<Value> {
        validate_key(id)?;
        let mut cache = self.cache.lock().await;
        // Always obtain current authoritative state under the lock; never publish
        // a stale streaming payload after a concurrent cancellation has settled.
        let response = app
            .client
            .post(app.route())
            .bearer_auth(app.config.sessions.token()?)
            .json(
                &json!({"jsonrpc":"2.0","id":"projection","method":"tasks/get","params":{"id":id}}),
            )
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(|_| Fault::new("projection_unavailable", "central task lookup unavailable"))?;
        let value = body(response).await?;
        let mut task = value["result"].clone();
        let context = task["contextId"]
            .as_str()
            .ok_or_else(|| Fault::invalid("central task lacks context"))?;
        validate_key(context)?;
        if task["id"] != id || task["kind"] != "task" || task["metadata"]["akh"] != app.config.akh {
            return Err(Fault::conflict("central task projection identity differs"));
        }
        let response = app
            .client
            .get(
                app.config
                    .sessions
                    .route(&format!("/v1/sessions/{}", segment(context))),
            )
            .bearer_auth(app.config.sessions.token()?)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(|_| {
                Fault::new(
                    "projection_unavailable",
                    "central context lookup unavailable",
                )
            })?;
        let session = body(response).await?;
        if session["id"] != context || session["akh"] != app.config.akh {
            return Err(Fault::conflict(
                "central context projection identity differs",
            ));
        }
        let external = session["conversation_key"]
            .as_str()
            .and_then(|key| key.strip_prefix(&self.context_prefix()))
            .unwrap_or(context)
            .to_owned();
        let central = context.to_owned();
        map_context(&mut task, &external, &central);
        let context = external.as_str();
        let encoded =
            serde_json::to_vec(&task).map_err(|_| Fault::invalid("task serialization"))?;
        let fingerprint = akh_session_services::digest(&encoded);
        if cache
            .get(id)
            .is_some_and(|(old, at)| old == &fingerprint && at.elapsed() < Duration::from_secs(30))
        {
            return Ok(task);
        }
        self.ensure_session(context, true).await?;
        let response = self
            .request(reqwest::Method::POST, "/api/tasks")?
            .json(&task)
            .send()
            .await
            .map_err(|_| Fault::new("projection_unavailable", "kagent task save unconfirmed"))?;
        let value = body(response).await?;
        if value["error"] != false
            || value["data"]["id"] != id
            || value["data"]["contextId"] != context
        {
            return Err(Fault::conflict(
                "kagent task save did not acknowledge the central identity",
            ));
        }
        // Bounded optimization only. Losing this cache repeats idempotent REST
        // projections; it never repeats a central message or workspace effect.
        if cache.len() >= 10000 {
            cache.clear();
        }
        cache.insert(id.into(), (fingerprint, tokio::time::Instant::now()));
        Ok(task)
    }
    pub async fn refresh(&self, app: &App) -> Result<()> {
        let response = app
            .client
            .get(app.config.sessions.route("/v1/sessions"))
            .bearer_auth(app.config.sessions.token()?)
            .timeout(Duration::from_secs(20))
            .send()
            .await
            .map_err(|_| {
                Fault::new(
                    "projection_unavailable",
                    "central session listing unavailable",
                )
            })?;
        let sessions = body(response).await?;
        let sessions: Vec<SessionRecord> = serde_json::from_value(sessions)
            .map_err(|_| Fault::invalid("invalid central session projection source"))?;
        // The mounted central credential must be dedicated to this facade/user.
        let mut failure = None;
        for session in sessions.into_iter().filter(|s| s.akh == app.config.akh) {
            for turn in session.turns {
                if let Err(error) = self.publish(app, &turn.id).await {
                    // Upstream soft-deleted task IDs stay burned. A failed view
                    // must not prevent other central tasks from being projected.
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

pub(super) fn map_context(result: &mut Value, external: &str, central: &str) {
    result["contextId"] = json!(external);
    if !result["metadata"].is_object() {
        result["metadata"] = json!({});
    }
    result["metadata"]["akhSessionId"] = json!(central);
    if let Some(history) = result["history"].as_array_mut() {
        for message in history {
            if message.get("contextId").is_some() {
                message["contextId"] = json!(external);
            }
        }
    }
    if result["status"]["message"].get("contextId").is_some() {
        result["status"]["message"]["contextId"] = json!(external);
    }
}

pub(super) fn task_id(value: &Value) -> Option<&str> {
    let result = &value["result"];
    match result["kind"].as_str() {
        Some("task") => result["id"].as_str(),
        Some("status-update" | "artifact-update") => result["taskId"].as_str(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tower::ServiceExt;
    #[tokio::test]
    async fn projection_retries_current_state_and_rejects_another_user_before_admission() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "test-integration-token-0123456789").unwrap();
        let fail = Arc::new(AtomicBool::new(true));
        let cancelled = Arc::new(AtomicBool::new(false));
        let saved = Arc::new(tokio::sync::Mutex::new(Vec::<Value>::new()));
        let c = cancelled.clone();
        let f = fail.clone();
        let s = saved.clone();
        let routes = Router::new()
            .route("/a2a/test", post(move |headers:HeaderMap, Json(request):Json<Value>| {
                let c=c.clone(); async move {
                    assert_eq!(headers[header::AUTHORIZATION], "Bearer test-integration-token-0123456789");
                    assert!(headers.get("x-user-id").is_none());
                    // Publishing is read-only against central task authority.
                    assert_eq!(request["method"],"tasks/get");
                    Json(json!({"result":{"kind":"task","id":"turn","contextId":"session",
                        "status":{"state":if c.load(Ordering::SeqCst){"canceled"}else{"submitted"}},"metadata":{"akh":"test"}}}))
                }
            }))
            .route("/v1/sessions/session",get(||async{Json(json!({"id":"session","akh":"test","conversation_key":null}))}))
            .route("/api/sessions/session",get(||async{StatusCode::NOT_FOUND}))
            .route("/api/sessions",post(|headers:HeaderMap,Json(value):Json<Value>|async move{
                assert_eq!(headers["x-user-id"],"owner");
                assert_eq!(headers["x-agent-name"],"ns/agent");
                assert_eq!(value["id"],"session");
                Json(json!({"data":{"id":"session","user_id":"owner","agent_id":"ns__NS__agent"}}))
            }))
            .route("/api/tasks",post(move |Json(task):Json<Value>| {
                let f=f.clone();let s=s.clone();async move {
                    if f.swap(false,Ordering::SeqCst) {return (StatusCode::BAD_GATEWAY,Json(json!({"error":true})));}
                    s.lock().await.push(task.clone());
                    (StatusCode::CREATED,Json(json!({"error":false,"data":task})))
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, routes).await.unwrap() });
        let endpoint = Endpoint {
            token_env: None,
            url,
            token_file: token,
            ca_file: None,
            allow_insecure_http: false,
            request_timeout_secs: None,
            anonymous: false,
        };
        let config = super::super::Config {
            listen: "127.0.0.1:0".parse().unwrap(),
            sessions: endpoint.clone(),
            akh: "test".into(),
            public_url: "http://facade/".into(),
            trust_kagent_network: true,
            projection: Some(Config {
                controller: endpoint,
                agent_ref: "ns/agent".into(),
                user_id: "owner".into(),
                workspace_profile: Some("test".into()),
            }),
        };
        let app = App::new(config).unwrap();
        let projection = app.projection.as_ref().unwrap();
        assert!(projection.publish(&app, "turn").await.is_err());
        cancelled.store(true, Ordering::SeqCst);
        projection.publish(&app, "turn").await.unwrap();
        projection.publish(&app, "turn").await.unwrap();
        let values = saved.lock().await;
        assert_eq!(values.len(), 1);
        assert_eq!(values[0]["status"]["state"], "canceled");
        drop(values);
        // Heal an upstream snapshot overwritten by a retiring rolling replica,
        // even when the central task has not changed again.
        projection.cache.lock().await.get_mut("turn").unwrap().1 =
            tokio::time::Instant::now() - Duration::from_secs(31);
        projection.publish(&app, "turn").await.unwrap();
        assert_eq!(saved.lock().await.len(), 2);
        for user in [None, Some("other-owner")] {
            let mut request =
                axum::http::Request::post("/").header("content-type", "application/json");
            if let Some(user) = user {
                request = request.header("x-user-id", user);
            }
            let response = super::super::router(app.clone())
                .oneshot(request.body(Body::from("{}")).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
        }
        server.abort();
    }
}

#[cfg(test)]
mod context_tests {
    use super::*;
    #[test]
    fn alias_changes_routing_without_rewriting_user_or_tool_payloads() {
        let mut event = json!({"kind":"task","id":"turn","contextId":"central",
            "history":[{"contextId":"central","parts":[{"kind":"data","data":{"contextId":"user-value"}}]}],
            "status":{"message":{"contextId":"central","parts":[{"kind":"data","data":{"contextId":"tool-value"}}]}},
            "artifacts":[{"parts":[{"kind":"data","data":{"contextId":"artifact-value"}}]}]});
        let before = event.clone();
        map_context(&mut event, "external", "central");
        assert_eq!(event["contextId"], "external");
        assert_eq!(event["metadata"]["akhSessionId"], "central");
        assert_eq!(event["history"][0]["parts"], before["history"][0]["parts"]);
        assert_eq!(
            event["status"]["message"]["parts"],
            before["status"]["message"]["parts"]
        );
        assert_eq!(event["artifacts"], before["artifacts"]);
    }
}
