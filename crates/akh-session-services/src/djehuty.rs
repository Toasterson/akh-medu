//! Native Djehuty thread pages. The configured bearer owns the Board identity;
//! this reader sends no agent override, writes no Board messages and acknowledges no inbox.
use crate::{
    config::DjehutySource,
    http::{Client, segment},
    ingress::Events,
    store::key,
    types::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Serialize, Deserialize)]
struct Cursor {
    endpoint: String,
    thread: String,
    message_id: Option<String>,
    position: Option<(u64, String)>,
}

fn uuid(value: &str) -> Result<String> {
    uuid::Uuid::parse_str(value)
        .map(|id| id.to_string())
        .map_err(|_| Fault::invalid("invalid Djehuty UUID"))
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| Fault::invalid(format!("Djehuty message lacks {key}")))
}
fn timestamp(value: &str) -> Result<u64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map_err(|_| Fault::invalid("Djehuty timestamp is not RFC3339"))?
        .timestamp_millis()
        .try_into()
        .map_err(|_| Fault::invalid("Djehuty timestamp precedes epoch"))
}

pub async fn poll(service: &Events, source: &DjehutySource, thread: &str) -> Result<()> {
    validate_key(&source.source)?;
    let thread = uuid(thread)?;
    let endpoint = service.config.endpoint(&source.endpoint)?;
    let k = key("djehuty_cursor", &[&source.source, &thread]);
    let mut cursor = service.store.get::<Cursor>(&k)?.unwrap_or(Cursor {
        endpoint: endpoint.url.clone(),
        thread: thread.clone(),
        message_id: None,
        position: None,
    });
    if cursor.endpoint != endpoint.url || cursor.thread != thread {
        return Err(Fault::conflict(
            "Djehuty source was rebound; use a new source ID instead of reusing its cursor",
        ));
    }
    if source.page_size == 0 || source.page_size > 200 {
        return Err(Fault::invalid("Djehuty page size must be 1..200"));
    }
    let mut path = format!(
        "/v1/threads/{}/messages?limit={}",
        segment(&thread),
        source.page_size
    );
    if let Some(after) = &cursor.message_id {
        path.push_str(&format!("&after_message_id={}", segment(after)));
    }
    let client = Client::new(service.config.clone())?;
    let (messages, headers): (Vec<Value>, _) = client
        .call_with_headers(&source.endpoint, reqwest::Method::GET, &path, None)
        .await?;
    if messages.len() > source.page_size as usize {
        return Err(Fault::invalid("Djehuty page exceeds advertised limit"));
    }
    let next = headers
        .get("next-cursor")
        .map(|v| {
            v.to_str()
                .map_err(|_| Fault::invalid("invalid Djehuty next-cursor header"))
        })
        .transpose()?
        .map(uuid)
        .transpose()?;
    let mut prepared = vec![];
    let ignored = source
        .ignore_authors
        .iter()
        .map(|id| uuid(id))
        .collect::<Result<Vec<_>>>()?;
    for message in messages {
        let id = uuid(text(&message, "id")?)?;
        let author = uuid(text(&message, "author_id")?)?;
        if uuid(text(&message, "thread_id")?)? != thread {
            return Err(Fault::conflict(
                "Djehuty returned a message from another thread",
            ));
        }
        let at = timestamp(text(&message, "created_at")?)?;
        let position = (at, id.clone());
        if cursor.position.as_ref().is_some_and(|old| old >= &position)
            || cursor.message_id.as_ref() == Some(&id)
        {
            return Err(Fault::conflict(
                "Djehuty cursor did not advance in timestamp/ID order",
            ));
        }
        text(&message, "summary")?;
        text(&message, "message_type")?;
        if message.get("payload").is_none() {
            return Err(Fault::invalid("Djehuty message has no payload"));
        }
        let expired = message
            .get("expires_at")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| Fault::invalid("invalid Djehuty expiry"))
                    .and_then(timestamp)
            })
            .transpose()?
            .is_some_and(|at| at <= now_ms());
        let reason = if ignored.contains(&author) {
            Some("configured author filter")
        } else if expired {
            Some("expired at intake")
        } else {
            None
        };
        prepared.push((
            EventEnvelope {
                schema_version: SCHEMA_VERSION,
                id: format!("{thread}/{id}"),
                source: source.source.clone(),
                event_type: "djehuty.message".into(),
                occurred_at: at,
                subject: Some(id.clone()),
                correlation_id: Some(thread.clone()),
                data: message,
            },
            reason,
        ));
        cursor.message_id = Some(id);
        cursor.position = Some(position);
    }
    if next.is_some() && (prepared.is_empty() || next != cursor.message_id) {
        return Err(Fault::conflict(
            "Djehuty next cursor does not identify the final complete message",
        ));
    }
    for (event, reason) in prepared {
        if let Some(reason) = reason {
            service.store.transaction(|tx| {
                tx.put(
                    &key("djehuty_ignored", &[&source.source, &event.id]),
                    &json!({"reason":reason,"event":event}),
                )
            })?;
        } else {
            service.ingest(&service.source_principal(&source.source), event)?;
        }
    }
    // Committing the cursor last makes a crash replay only stable, already durable IDs.
    service.store.transaction(|tx| tx.put(&k, &cursor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, ingress::IngressReceipt, store::Store};
    use axum::{
        Json, Router,
        extract::{Query, State},
        http::HeaderMap,
        routing::get,
    };
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };
    const THREAD: &str = "00000000-0000-7000-8000-000000000001";
    const AUTHOR: &str = "00000000-0000-7000-8000-000000000002";
    const FIRST: &str = "00000000-0000-7000-8000-000000000003";
    const LAST: &str = "00000000-0000-7000-8000-000000000004";
    #[derive(Default)]
    struct Board {
        requests: Vec<Option<String>>,
        wrong_thread: bool,
        force_replay: bool,
    }
    async fn page(
        State(state): State<Arc<Mutex<Board>>>,
        headers: HeaderMap,
        Query(query): Query<HashMap<String, String>>,
    ) -> (HeaderMap, Json<Vec<Value>>) {
        assert_eq!(
            headers["authorization"],
            "Bearer fixture-djehuty-owned-token-0123456789"
        );
        assert!(!query.contains_key("agent_id"));
        let mut state = state.lock().unwrap();
        state.requests.push(query.get("after_message_id").cloned());
        if query.get("after_message_id").is_some() && !state.force_replay {
            return (HeaderMap::new(), Json(vec![]));
        }
        let thread = if state.wrong_thread { AUTHOR } else { THREAD };
        let messages=[FIRST,LAST].iter().map(|id|json!({"id":id,"thread_id":thread,"author_id":AUTHOR,"created_at":"2026-09-12T00:00:00Z","message_type":"note.v1","summary":"fixture","payload":{"instructions":"untrusted text"}})).collect();
        let mut headers = HeaderMap::new();
        headers.insert("next-cursor", LAST.parse().unwrap());
        (headers, Json(messages))
    }
    #[tokio::test]
    async fn native_pages_are_durable_before_cursor_and_never_override_board_identity() {
        let board = Arc::new(Mutex::new(Board::default()));
        let app = Router::new()
            .route("/v1/threads/{id}/messages", get(page))
            .with_state(board.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "fixture-djehuty-owned-token-0123456789").unwrap();
        let config:Config=serde_json::from_value(json!({"service_id":"events","listen":"127.0.0.1:0","data_dir":dir.path(),"principals":[{"id":"owner","token_file":token,"roles":["admin"]}],"endpoints":{"board":{"url":url,"token_file":token}},"routes":[{"id":"thread","source":"owned-board","event_type":"djehuty.message","automation":"review"}]})).unwrap();
        let service = Events::new(
            Arc::new(config),
            Store::open(&dir.path().join("events.redb")).unwrap(),
        )
        .unwrap();
        let source = DjehutySource {
            endpoint: "board".into(),
            source: "owned-board".into(),
            threads: vec![THREAD.into()],
            ignore_authors: vec![],
            interval_secs: 10,
            page_size: 50,
        };
        poll(&service, &source, THREAD).await.unwrap();
        poll(&service, &source, THREAD).await.unwrap();
        assert_eq!(
            board.lock().unwrap().requests,
            vec![None, Some(LAST.into())]
        );
        let receipts = service.store.scan::<IngressReceipt>("ingress/").unwrap();
        assert_eq!(receipts.len(), 2);
        assert_eq!(
            receipts[0].event.data["payload"]["instructions"],
            "untrusted text"
        );
        assert_eq!(receipts[0].event.correlation_id.as_deref(), Some(THREAD));
        assert_eq!(
            service
                .store
                .scan::<crate::ingress::Delivery>("delivery/")
                .unwrap()
                .len(),
            2
        );
        // Emulate a crash after durable ingress but before advancing the watermark.
        service
            .store
            .transaction(|tx| tx.delete(&key("djehuty_cursor", &[&source.source, THREAD])))
            .unwrap();
        poll(&service, &source, THREAD).await.unwrap();
        assert_eq!(
            service
                .store
                .scan::<IngressReceipt>("ingress/")
                .unwrap()
                .len(),
            2
        );
        board.lock().unwrap().force_replay = true;
        assert!(
            poll(&service, &source, THREAD)
                .await
                .unwrap_err()
                .message
                .contains("did not advance")
        );
        let mut another = source.clone();
        another.source = "new-source".into();
        board.lock().unwrap().wrong_thread = true;
        assert!(
            poll(&service, &another, THREAD)
                .await
                .unwrap_err()
                .message
                .contains("another thread")
        );
        assert!(
            service
                .store
                .get::<Cursor>(&key("djehuty_cursor", &[&another.source, THREAD]))
                .unwrap()
                .is_none()
        );
        server.abort();
    }
}
