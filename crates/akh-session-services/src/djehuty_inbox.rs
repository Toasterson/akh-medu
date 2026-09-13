//! An Akh's own inbox becomes durable events before acknowledgement. The pending
//! Board API is deliberately non-consuming; never fall back to read-and-mark-read.
use crate::{
    config::DjehutyInbox,
    djehuty::{text, timestamp, uuid},
    http::{Client, segment},
    ingress::Events,
    store::key,
    types::*,
};
use serde_json::{Value, json};

pub async fn poll(service: &Events, source: &DjehutyInbox) -> Result<()> {
    let recipient = uuid(&source.recipient_id)?;
    let endpoint = service.config.endpoint(&source.endpoint)?;
    let client = Client::new(service.config.clone())?;
    // Verify the mounted credential, without an identity override or token logging.
    let identity: Value = client.get(&source.endpoint, "/v1/agents/me").await?;
    if uuid(text(&identity, "id")?)? != recipient {
        return Err(Fault::new(
            "forbidden",
            "Djehuty inbox credential belongs to another identity",
        ));
    }
    let binding_key = key("djehuty_inbox_binding", &[&source.source]);
    let binding = json!({"endpoint":endpoint.url,"recipient_id":recipient});
    service.store.transaction(|tx| {
        if tx
            .get::<Value>(&binding_key)?
            .is_some_and(|old| old != binding)
        {
            return Err(Fault::conflict(
                "Djehuty inbox source was rebound; use a new source ID",
            ));
        }
        tx.put(&binding_key, &binding)
    })?;
    let messages: Vec<Value> = client
        .get(
            &source.endpoint,
            &format!("/v1/inbox/pending?limit={}", source.page_size),
        )
        .await?;
    if messages.len() > source.page_size as usize {
        return Err(Fault::invalid(
            "Djehuty inbox page exceeds configured limit",
        ));
    }
    let mut accepted = 0;
    let mut ignored = 0;
    for item in messages {
        let mut message = item["message"].clone();
        let id = uuid(text(&message, "id")?)?;
        let author = uuid(text(&message, "author_id")?)?;
        let at = timestamp(text(&message, "created_at")?)?;
        let kind = text(&message, "message_type")?.to_owned();
        text(&message, "summary")?;
        if !message["thread_id"].is_null() || !message["payload"].is_object() {
            return Err(Fault::invalid(
                "Djehuty pending item is not an inbox message",
            ));
        }
        let expired = message
            .get("expires_at")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| Fault::invalid("invalid inbox expiry"))
                    .and_then(timestamp)
            })
            .transpose()?
            .is_some_and(|expiry| expiry <= now_ms());
        let reason = if author == recipient {
            Some("self-authored")
        } else if !source.allowed_authors.is_empty() && !source.allowed_authors.contains(&author) {
            Some("author not granted")
        } else if !source.message_types.contains(&kind) {
            Some("protocol does not wake this Akh")
        } else if kind == "coordination/delegation@v1"
            && message["payload"]["assignee_id"].as_str() != Some(&recipient)
        {
            Some("delegation names another assignee")
        } else if expired {
            Some("expired at intake")
        } else {
            None
        };
        message["recipient_id"] = recipient.clone().into();
        let event = EventEnvelope {
            schema_version: SCHEMA_VERSION,
            id: format!("inbox/{recipient}/{id}"),
            source: source.source.clone(),
            event_type: "djehuty.inbox".into(),
            occurred_at: at,
            subject: Some(id.clone()),
            correlation_id: Some(recipient.clone()),
            data: message,
        };
        if let Some(reason) = reason {
            service.store.transaction(|tx| {
                tx.put(
                    &key("djehuty_ignored", &[&source.source, &event.id]),
                    &json!({"reason":reason,"event":event}),
                )
            })?;
            ignored += 1;
        } else {
            let receipt = service.ingest(&service.source_principal(&source.source), event)?;
            if receipt.deliveries.is_empty() {
                return Err(Fault::invalid(
                    "Djehuty inbox has no configured session route",
                ));
            }
            accepted += 1;
        }
        // An outage here replays the same immutable event through ingress deduplication.
        // Acknowledgement means durable intake, not model completion or task success.
        let ack: Value = client
            .post(
                &source.endpoint,
                &format!("/v1/inbox/messages/{}/acknowledge", segment(&id)),
                &json!({}),
            )
            .await?;
        if ack["message"]["id"].as_str() != Some(&id) || ack["state"] != "acknowledged" {
            return Err(Fault::conflict(
                "Djehuty acknowledged a different inbox delivery",
            ));
        }
    }
    service.store.transaction(|tx| {
        tx.put(
            &key("djehuty_inbox_status", &[&source.source]),
            &json!({"recipient_id":recipient,"at":now_ms(),"accepted":accepted,"ignored":ignored}),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, ingress::IngressReceipt, store::Store};
    use axum::{
        Json, Router,
        extract::State,
        http::{HeaderMap, StatusCode},
        routing::{get, post},
    };
    use std::sync::{Arc, Mutex};
    const RECIPIENT: &str = "00000000-0000-7000-8000-000000000001";
    const AUTHOR: &str = "00000000-0000-7000-8000-000000000002";
    const MESSAGE: &str = "00000000-0000-7000-8000-000000000003";
    struct Board {
        identity: String,
        message: Value,
        fail_ack: bool,
        acknowledged: bool,
        polls: usize,
    }
    async fn me(State(board): State<Arc<Mutex<Board>>>, headers: HeaderMap) -> Json<Value> {
        assert_eq!(
            headers["authorization"],
            "Bearer fixture-own-inbox-token-0123456789"
        );
        assert!(!headers.contains_key("x-agent-id"));
        Json(json!({"id":board.lock().unwrap().identity}))
    }
    async fn pending(State(board): State<Arc<Mutex<Board>>>) -> Json<Vec<Value>> {
        let mut b = board.lock().unwrap();
        b.polls += 1;
        Json(if b.acknowledged {
            vec![]
        } else {
            vec![json!({"message":b.message,"state":"unread"})]
        })
    }
    async fn ack(
        State(board): State<Arc<Mutex<Board>>>,
    ) -> std::result::Result<Json<Value>, StatusCode> {
        let mut b = board.lock().unwrap();
        if b.fail_ack {
            return Err(StatusCode::SERVICE_UNAVAILABLE);
        }
        b.acknowledged = true;
        Ok(Json(json!({"message":b.message,"state":"acknowledged"})))
    }
    #[tokio::test]
    async fn durable_before_ack_replay_and_identity_are_enforced() {
        let board = Arc::new(Mutex::new(Board {
            identity: RECIPIENT.into(),
            message: json!({"id":MESSAGE,"author_id":AUTHOR,"thread_id":null,"reply_to":null,
                "message_type":"coordination/request@v1","payload":{"objective":"Report availability"},
                "summary":"Ping","created_at":chrono::DateTime::from_timestamp_millis(now_ms() as i64).unwrap().to_rfc3339(),"expires_at":null}),
            fail_ack: true,
            acknowledged: false,
            polls: 0,
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new()
            .route("/v1/agents/me", get(me))
            .route("/v1/inbox/pending", get(pending))
            .route(
                &format!("/v1/inbox/messages/{MESSAGE}/acknowledge"),
                post(ack),
            )
            .with_state(board.clone());
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "fixture-own-inbox-token-0123456789").unwrap();
        let source: DjehutyInbox=serde_json::from_value(json!({"endpoint":"board","source":"owned-inbox","recipient_id":RECIPIENT,"message_types":["coordination/request@v1"],"page_size":10})).unwrap();
        let config:Config=serde_json::from_value(json!({"service_id":"events","listen":"127.0.0.1:0","data_dir":dir.path(),"principals":[{"id":"operator","token_file":token,"roles":["admin"]}],"endpoints":{"board":{"url":url,"token_file":token}},"djehuty_inboxes":[source],"routes":[{"id":"wake","source":"owned-inbox","event_type":"djehuty.inbox","automation":"coordinate"}]})).unwrap();
        config.validate().unwrap();
        let service = Events::new(
            Arc::new(config),
            Store::open(&dir.path().join("events.redb")).unwrap(),
        )
        .unwrap();
        assert!(poll(&service, &source).await.is_err());
        assert!(!board.lock().unwrap().acknowledged);
        assert_eq!(
            service
                .store
                .scan::<IngressReceipt>("ingress/")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            service
                .store
                .scan::<crate::ingress::Delivery>("delivery/")
                .unwrap()
                .len(),
            1
        );
        board.lock().unwrap().fail_ack = false;
        poll(&service, &source).await.unwrap();
        assert!(board.lock().unwrap().acknowledged);
        assert_eq!(
            service
                .store
                .scan::<IngressReceipt>("ingress/")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            service
                .store
                .scan::<crate::ingress::Delivery>("delivery/")
                .unwrap()
                .len(),
            1
        );
        let polls = board.lock().unwrap().polls;
        board.lock().unwrap().identity = AUTHOR.into();
        assert_eq!(poll(&service, &source).await.unwrap_err().code, "forbidden");
        assert_eq!(board.lock().unwrap().polls, polls);
        board.lock().unwrap().identity = RECIPIENT.into();
        board.lock().unwrap().acknowledged = false;
        board.lock().unwrap().message["id"] = "00000000-0000-7000-8000-000000000004".into();
        board.lock().unwrap().message["author_id"] = RECIPIENT.into();
        // Filtered content is preserved without a model wake; the fixed ack route above
        // intentionally fails, proving it will remain pending until acknowledged.
        assert!(poll(&service, &source).await.is_err());
        assert_eq!(
            service
                .store
                .scan::<Value>("djehuty_ignored/")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            service
                .store
                .scan::<crate::ingress::Delivery>("delivery/")
                .unwrap()
                .len(),
            1
        );
        server.abort();
    }
}
