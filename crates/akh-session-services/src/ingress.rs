//! Deterministic event ingress and transactional routing outbox. No model or session clock.
use crate::{
    config::{Config, Principal},
    http::Client,
    store::{Store, key},
    types::*,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delivery {
    pub id: String,
    pub route: String,
    pub automation: String,
    pub request: InvokeAutomation,
    pub attempts: u32,
    pub next_attempt: u64,
    pub admission: Option<Admission>,
    pub error: Option<Fault>,
    pub terminal: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngressReceipt {
    pub event: EventEnvelope,
    pub deliveries: Vec<String>,
}
#[derive(Clone)]
pub struct Events {
    pub config: Arc<Config>,
    pub store: Store,
    client: Client,
    lock: Arc<tokio::sync::Mutex<()>>,
}
impl Events {
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        Ok(Self {
            client: Client::new(config.clone())?,
            config,
            store,
            lock: Default::default(),
        })
    }
    pub fn ingest(&self, principal: &Principal, event: EventEnvelope) -> Result<IngressReceipt> {
        if !principal.has_role("event") && !principal.has_role("scheduler") {
            return Err(Fault::new("forbidden", "event or scheduler role required"));
        }
        if !principal.has_role("admin") && !principal.sources.contains(&event.source) {
            return Err(Fault::new("forbidden", "event source not granted"));
        }
        if event.schema_version != SCHEMA_VERSION {
            return Err(Fault::invalid("unsupported event schema"));
        }
        validate_key(&event.source)?;
        validate_key(&event.id)?;
        validate_key(&event.event_type)?;
        if serde_json::to_vec(&event)
            .map_err(|e| Fault::invalid(e.to_string()))?
            .len()
            > 1024 * 1024
        {
            return Err(Fault::invalid("event exceeds 1 MiB"));
        }
        self.store.transaction(|tx| {
            let k = key("ingress", &[&event.source, &event.id]);
            if let Some(old) = tx.get::<IngressReceipt>(&k)? {
                if old.event != event {
                    return Err(Fault::conflict(
                        "source event ID was reused with a different payload",
                    ));
                }
                return Ok(old);
            }
            let mut deliveries = vec![];
            for route in &self.config.routes {
                if route.source != event.source || route.event_type != event.event_type {
                    continue;
                }
                let id = crate::digest(
                    key("delivery", &[&event.source, &event.id, &route.id]).as_bytes(),
                );
                let delivery = Delivery {
                    id: id.clone(),
                    route: route.id.clone(),
                    automation: route.automation.clone(),
                    request: InvokeAutomation {
                        bootstrap_requirements: None,
                        occurrence_id: id.clone(),
                        event: event.clone(),
                    },
                    attempts: 0,
                    next_attempt: now_ms(),
                    admission: None,
                    error: None,
                    terminal: false,
                };
                tx.put(&key("delivery", &[&id]), &delivery)?;
                deliveries.push(id);
            }
            let receipt = IngressReceipt { event, deliveries };
            tx.put(&k, &receipt)?;
            Ok(receipt)
        })
    }
    fn coalesce_pending(&self) -> Result<()> {
        self.store.transaction(|tx| {
            let deliveries = tx.scan::<Delivery>("delivery/")?;
            for route in self.config.routes.iter().filter(|r| r.coalesce_pending) {
                let Some(latest) = deliveries.iter().filter(|d| d.route == route.id && (!d.terminal || d.admission.is_some()))
                    .max_by_key(|d| (d.request.event.occurred_at, d.id.clone())) else { continue; };
                for old in deliveries.iter().filter(|d| d.route == route.id && d.id != latest.id && !d.terminal && d.admission.is_none()) {
                    // A transport failure may hide an admitted turn. Only
                    // supersede never-attempted or explicitly refused work.
                    if old.attempts != 0 && !old.error.as_ref().is_some_and(|e| e.code == "busy") { continue; }
                    let mut old = old.clone();
                    old.terminal = true;
                    old.error = Some(Fault::new("superseded", format!("newer state notification {}; original event retained, no turn inferred", latest.id)));
                    tx.put(&key("delivery", &[&old.id]), &old)?;
                }
            }
            Ok(())
        })
    }
    pub async fn drain(&self) -> Result<()> {
        let _lock = self.lock.lock().await;
        self.coalesce_pending()?;
        for mut delivery in self
            .store
            .scan::<Delivery>("delivery/")?
            .into_iter()
            .filter(|d| !d.terminal && d.next_attempt <= now_ms())
            .take(64)
        {
            let reply: Result<Admission> = self
                .client
                .post(
                    "sessions",
                    &format!(
                        "/v1/automations/{}/invoke",
                        crate::http::segment(&delivery.automation)
                    ),
                    &delivery.request,
                )
                .await;
            delivery.attempts = delivery.attempts.saturating_add(1);
            match reply {
                Ok(admission) => {
                    delivery.admission = Some(admission);
                    delivery.terminal = true;
                    delivery.error = None;
                }
                Err(error) => {
                    delivery.terminal = matches!(
                        error.code.as_str(),
                        "forbidden" | "invalid_request" | "conflict" | "expired" | "not_found"
                    );
                    delivery.next_attempt = now_ms() + (1000u64 << delivery.attempts.min(8));
                    delivery.error = Some(error);
                }
            }
            self.store
                .transaction(|tx| tx.put(&key("delivery", &[&delivery.id]), &delivery))?;
        }
        Ok(())
    }
    pub fn sources(&self) {
        for source in self.config.djehuty_inboxes.clone() {
            if !source.enabled {
                continue;
            }
            let service = self.clone();
            tokio::spawn(async move {
                loop {
                    if let Err(error) = crate::djehuty_inbox::poll(&service, &source).await {
                        tracing::warn!(source=%source.source,code=%error.code,"Djehuty inbox ingestion deferred");
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(source.interval_secs.max(1)))
                        .await;
                }
            });
        }
        for source in self.config.djehuty.clone() {
            for thread in source.threads.clone() {
                let service = self.clone();
                let source = source.clone();
                tokio::spawn(async move {
                    loop {
                        if let Err(error) = crate::djehuty::poll(&service, &source, &thread).await {
                            tracing::warn!(source=%source.source,thread=%thread,code=%error.code,"Djehuty ingestion deferred");
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(
                            source.interval_secs.max(1),
                        ))
                        .await;
                    }
                });
            }
        }
        for source in self.config.brokers.clone() {
            let service = self.clone();
            tokio::spawn(async move {
                loop {
                    if let Err(error) = service.consume(&source).await {
                        tracing::warn!(source=%source.source,code=%error.code,"event broker disconnected");
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            });
        }
        for source in self.config.boards.clone() {
            let service = self.clone();
            tokio::spawn(async move {
                loop {
                    if let Err(error) = service.poll_board(&source).await {
                        tracing::warn!(source=%source.source,code=%error.code,"board event poll failed");
                    }
                    tokio::time::sleep(std::time::Duration::from_secs(source.interval_secs.max(1)))
                        .await;
                }
            });
        }
    }
    pub(crate) fn source_principal(&self, source: &str) -> Principal {
        Principal {
            id: self.config.service_id.clone(),
            token_file: "/unused".into(),
            token_sha256: false,
            roles: vec!["event".into()],
            akhs: vec![],
            sources: vec![source.into()],
        }
    }
    async fn consume(&self, source: &crate::config::BrokerSource) -> Result<()> {
        use futures_util::StreamExt;
        use lapin::{Connection, ConnectionProperties, options::*, types::FieldTable};
        let uri = std::fs::read_to_string(&source.uri_file)
            .map_err(|_| Fault::new("auth_unavailable", "broker URI file unavailable"))?;
        let properties = ConnectionProperties::default()
            .with_executor(tokio_executor_trait::Tokio::current())
            .with_reactor(tokio_reactor_trait::Tokio);
        let connection = Connection::connect(uri.trim(), properties)
            .await
            .map_err(|_| Fault::new("transport", "broker connection failed"))?;
        let channel = connection
            .create_channel()
            .await
            .map_err(|_| Fault::new("transport", "broker channel failed"))?;
        channel
            .basic_qos(16, BasicQosOptions::default())
            .await
            .map_err(|_| Fault::new("transport", "broker QoS failed"))?;
        let mut consumer = channel
            .basic_consume(
                &source.queue,
                &self.config.service_id,
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await
            .map_err(|_| Fault::new("transport", "broker consume failed"))?;
        while let Some(delivery) = consumer.next().await {
            let delivery =
                delivery.map_err(|_| Fault::new("transport", "broker delivery failed"))?;
            let event: Result<EventEnvelope> = if delivery.data.len() <= 1024 * 1024 {
                serde_json::from_slice(&delivery.data)
                    .map_err(|_| Fault::invalid("invalid event envelope"))
            } else {
                Err(Fault::invalid("event too large"))
            };
            let result =
                event.and_then(|event| self.ingest(&self.source_principal(&source.source), event));
            match result {
                Ok(_) => {
                    delivery
                        .ack(BasicAckOptions::default())
                        .await
                        .map_err(|_| Fault::new("transport", "broker acknowledgement failed"))?;
                }
                Err(error) => {
                    let requeue = !matches!(
                        error.code.as_str(),
                        "invalid_request" | "forbidden" | "conflict"
                    );
                    delivery
                        .nack(BasicNackOptions {
                            multiple: false,
                            requeue,
                        })
                        .await
                        .map_err(|_| Fault::new("transport", "broker rejection failed"))?;
                    if requeue {
                        return Err(error);
                    }
                }
            }
        }
        Err(Fault::new("transport", "broker consumer ended"))
    }
    async fn poll_board(&self, source: &crate::config::BoardSource) -> Result<()> {
        #[derive(Deserialize)]
        struct Page {
            events: Vec<EventEnvelope>,
            next_cursor: String,
        }
        let cursor_key = key("board_cursor", &[&source.source, &source.endpoint]);
        let cursor = self.store.get::<String>(&cursor_key)?.unwrap_or_default();
        if !source.poll_path.starts_with('/') || source.poll_path.contains(['?', '#']) {
            return Err(Fault::invalid(
                "board poll path must be an absolute path without query or fragment",
            ));
        }
        let page: Page = self
            .client
            .get(
                &source.endpoint,
                &format!(
                    "{}?cursor={}",
                    source.poll_path,
                    crate::http::segment(&cursor)
                ),
            )
            .await?;
        for event in page.events {
            self.ingest(&self.source_principal(&source.source), event)?;
        }
        // A crash before the cursor commit replays only already durable source IDs.
        self.store
            .transaction(|tx| tx.put(&cursor_key, &page.next_cursor))
    }
}

#[cfg(test)]
mod coalescing_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn coalescing_preserves_uncertain_admissions_and_delegations() {
        let dir = tempfile::tempdir().unwrap();
        let config: Config = serde_json::from_value(json!({"service_id":"events","listen":"127.0.0.1:0","data_dir":dir.path(),"principals":[],"routes":[
            {"id":"clock","source":"clock","event_type":"tick","automation":"vesta","coalesce_pending":true},
            {"id":"inbox","source":"board","event_type":"delegation","automation":"vesta"}
        ]})).unwrap();
        let events = Events::new(
            Arc::new(config),
            Store::open(&dir.path().join("events.redb")).unwrap(),
        )
        .unwrap();
        let mut ids = vec![];
        for (source, kind, id, time) in [
            ("clock", "tick", "old", 1),
            ("clock", "tick", "uncertain", 2),
            ("clock", "tick", "new", 3),
            ("board", "delegation", "one", 1),
            ("board", "delegation", "two", 2),
        ] {
            let event=serde_json::from_value(json!({"schema_version":1,"source":source,"type":kind,"id":id,"occurred_at":time,"data":{}})).unwrap();
            ids.push(
                events
                    .ingest(&events.source_principal(source), event)
                    .unwrap()
                    .deliveries[0]
                    .clone(),
            );
        }
        events
            .store
            .transaction(|tx| {
                let k = key("delivery", &[&ids[1]]);
                let mut d: Delivery = tx.get(&k)?.unwrap();
                d.attempts = 1;
                d.error = Some(Fault::new("transport", "response lost"));
                tx.put(&k, &d)
            })
            .unwrap();
        events.coalesce_pending().unwrap();
        for (index, id) in ids.iter().enumerate() {
            let d: Delivery = events.store.get(&key("delivery", &[id])).unwrap().unwrap();
            assert_eq!(d.terminal, index == 0);
            if index == 0 {
                assert_eq!(d.error.unwrap().code, "superseded");
            }
        }
    }
}
