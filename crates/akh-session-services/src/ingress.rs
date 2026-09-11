//! Deterministic event ingress and transactional routing outbox. No model or session clock.
use crate::{
    config::{Config, Principal},
    http::Client,
    session::Admission,
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
                let id = crate::workspace::digest(
                    key("delivery", &[&event.source, &event.id, &route.id]).as_bytes(),
                );
                let delivery = Delivery {
                    id: id.clone(),
                    route: route.id.clone(),
                    automation: route.automation.clone(),
                    request: InvokeAutomation {
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
    pub async fn drain(&self) -> Result<()> {
        let _lock = self.lock.lock().await;
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
    fn source_principal(&self, source: &str) -> Principal {
        Principal {
            id: self.config.service_id.clone(),
            token_file: "/unused".into(),
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
