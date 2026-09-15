//! Combined native capture and real-core transport fixture. No GPU or daemon processes.
use super::*;
use akh_medu::nlu::vendored::quantized_qwen35::test_fixture::{
    TINY_REPLYING_TOKENIZER_JSON, tiny_gguf_replying,
};
use akh_runtime::Host;
use std::sync::atomic::AtomicBool;
use tower::ServiceExt;

struct HttpHost {
    router: Router,
    token: String,
    call: ToolCall,
    provider: Option<String>,
    hits: Vec<neural_spans::SpanHit>,
    delivered: usize,
}
impl HttpHost {
    fn post<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: Value,
    ) -> std::result::Result<T, String> {
        tokio::runtime::Handle::current().block_on(async {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("authorization", format!("Bearer {}", self.token))
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .map_err(|e| e.to_string())?;
            let response = self
                .router
                .clone()
                .oneshot(request)
                .await
                .map_err(|e| e.to_string())?;
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 65536)
                .await
                .map_err(|e| e.to_string())?;
            if !status.is_success() {
                return Err(format!("callback {path}: {status}"));
            }
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())
        })
    }
    fn call(&self, id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            ..self.call.clone()
        }
    }
}
impl Host for HttpHost {
    fn event(&mut self, event: RuntimeEvent) -> std::result::Result<(), String> {
        if let RuntimeEvent::Bound {
            provider_session, ..
        } = event
        {
            self.provider = Some(provider_session);
        }
        Ok(())
    }
    fn tool(&mut self, _: &str, _: &str, _: Value) -> std::result::Result<ToolResult, String> {
        Err("no tools in fixture".into())
    }
    fn approval(&mut self, _: &str, _: &str, _: &Value) -> std::result::Result<bool, String> {
        Err("no approvals in fixture".into())
    }
    fn question(&mut self, _: &str, _: &Value) -> std::result::Result<Value, String> {
        Err("no questions in fixture".into())
    }
    fn recall(
        &mut self,
        id: &str,
        space: &neural_spans::SpanSpace,
        vector: &[u8],
        limit: usize,
    ) -> std::result::Result<Option<Vec<neural_spans::SpanHit>>, String> {
        let reply: neural_spans::RecallReply = self.post(
            "/v1/runtime/recall",
            json!(neural_spans::RecallRequest {
                call: self.call(id, neural_spans::RecallRequest::CALL_NAME),
                space: space.clone(),
                vector: vector.to_vec(),
                limit,
            }),
        )?;
        if !reply.supported {
            return Err("real core unexpectedly lacks recall".into());
        }
        self.hits = reply.hits.clone();
        Ok(Some(reply.hits))
    }
    fn spans(
        &mut self,
        id: &str,
        captures: &[neural_spans::SpanCapture],
    ) -> std::result::Result<bool, String> {
        let reply: neural_spans::SpanDeliveryReply = self.post(
            "/v1/runtime/spans",
            json!(neural_spans::SpanDelivery {
                call: self.call(id, neural_spans::SpanDelivery::CALL_NAME),
                captures: captures.to_vec(),
            }),
        )?;
        if reply.accepted {
            self.delivered += captures.len();
        }
        Ok(reply.accepted)
    }
}

pub(super) async fn run(app: SessionApp, router: Router, token: &str, dir: &std::path::Path) {
    let model = tiny_gguf_replying(96, 8);
    let model_path = dir.join("combined-tiny.gguf");
    let tokenizer_path = dir.join("combined-tokenizer.json");
    std::fs::write(&model_path, &model).unwrap();
    std::fs::write(&tokenizer_path, TINY_REPLYING_TOKENIZER_JSON).unwrap();
    let profile: ProviderConfig = serde_json::from_value(json!({
        "revision":"1","kind":"internal","working_dir":dir,"binary":"/unused",
        "execution_mode":"brokered","timeout_secs":120,
        "native_candle":{"model":"fixture","model_path":model_path,"model_sha256":crate::digest(&model),
            "tokenizer_path":tokenizer_path,"tokenizer_sha256":crate::digest(TINY_REPLYING_TOKENIZER_JSON.as_bytes()),
            "device":{"kind":"cpu"},"max_context_tokens":96,"max_output_tokens":4,"max_iterations":4}
    })).unwrap();
    let owner = &app.sessions.config.principals[0];
    let session = app
        .sessions
        .create(
            owner,
            &NewSession {
                request_id: "native-combined".into(),
                akh: "test".into(),
                workspace_profile: "local".into(),
                conversation_key: None,
            },
        )
        .unwrap();
    let store = Store::open(&dir.join("native-combined.redb")).unwrap();
    let mut provider = None;
    let mut first_turn = String::new();
    let turns = [
        "Remember the word cedar",
        "One two three four five six seven eight nine ten eleven twelve",
        "Red blue green white black gray orange purple yellow silver gold brown",
        "North south east west up down left right near far above below",
        "What word?",
    ];
    for (index, text) in turns.iter().enumerate() {
        let turn = app
            .sessions
            .submit(
                owner,
                &session.id,
                &SubmitInput {
                    request_id: format!("native-{index}"),
                    text: (*text).into(),
                    artifacts: vec![],
                },
            )
            .unwrap();
        if index == 0 {
            first_turn = turn.id.clone();
        }
        app.sessions
            .store
            .transaction(|tx| {
                let mut current: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                current.turns[index].status = TurnStatus::Running;
                tx.put(&key("session", &[&session.id]), &current)
            })
            .unwrap();
        let mut binding = session.runtime.clone();
        binding.model = Some("fixture".into());
        let request: RuntimeJobRequest = serde_json::from_value(json!({
            "id":format!("native-job-{index}"),"session_id":session.id,"turn_id":turn.id,"generation":turn.generation,
            "binding":binding,"provider_session":provider,
            "workspace":{"id":"w","sandbox_id":"s","executor":"e","generation":1,"site":"test","root":"/remote-only","platform":"test","confinement":"test"},
            "input":turn.input,"instructions":"Answer.","tools":[],"approval_policy":"read_only"
        })).unwrap();
        let mut host = HttpHost {
            router: router.clone(),
            token: token.into(),
            call: ToolCall {
                id: "unused".into(),
                session_id: session.id.clone(),
                turn_id: turn.id.clone(),
                generation: turn.generation,
                name: "unused".into(),
                arguments: json!({}),
            },
            provider: None,
            hits: vec![],
            delivered: 0,
        };
        let (runtime_store, native_profile) = (store.clone(), profile.clone());
        let (output, host) = tokio::task::spawn_blocking(move || {
            let output = crate::internal::run(
                &runtime_store,
                &native_profile,
                &request,
                &mut host,
                &AtomicBool::new(false),
            )
            .unwrap();
            (output, host)
        })
        .await
        .unwrap();
        assert_eq!(output, "ok");
        assert_eq!(
            host.delivered, 1,
            "native capture must cross the session router"
        );
        if index == 0 {
            assert!(host.hits.is_empty());
        } else {
            assert_eq!(host.hits.len(), index);
            assert!(
                host.hits
                    .iter()
                    .any(|hit| hit.turn_id == first_turn && hit.text == "Remember the word cedar")
            );
        }
        provider = host.provider;
        if index == turns.len() - 1 {
            let history: Value = store
                .get(&key("internal", &[provider.as_deref().unwrap()]))
                .unwrap()
                .unwrap();
            let messages = history["messages"].as_array().unwrap();
            assert_eq!(
                messages[0]["content"], turns[0],
                "durable history must retain the evicted fact"
            );
            let recalled = &messages[messages.len() - 3];
            assert_eq!(
                recalled["role"], "system",
                "an evicted fact must be materialized before the last user turn"
            );
            assert!(
                recalled["content"].as_str().unwrap().contains(turns[0]),
                "recall-driven eviction must not lose a hit previously deduplicated as visible"
            );
        }
        app.sessions
            .store
            .transaction(|tx| {
                let mut current: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                current.turns[index].status = TurnStatus::Completed;
                current.turns[index].output = Some(output.clone());
                tx.put(&key("session", &[&session.id]), &current)
            })
            .unwrap();
        let completed = app.sessions.get(owner, &session.id).unwrap();
        app.publish_completions(&completed).await.unwrap();
    }
}
