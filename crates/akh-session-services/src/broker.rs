//! Session-scoped tool admission and durable downstream execution evidence.
use crate::{
    config::Principal,
    http::{Client, segment},
    session::Sessions,
    store::key,
    types::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRecord {
    pub call: ToolCall,
    pub operation_id: String,
    pub destination: ToolDestination,
    pub sent: bool,
    pub result: Option<ToolResult>,
    pub unknown: bool,
}
#[derive(Debug, Serialize)]
pub struct ToolPage {
    pub records: Vec<ToolRecord>,
    pub next_cursor: Option<String>,
}
fn record_key(call: &ToolCall) -> String {
    key("tool", &[&call.session_id, &call.turn_id, &call.id])
}
pub fn operation_id(call: &ToolCall) -> String {
    crate::digest(record_key(call).as_bytes())
}

#[derive(Clone)]
pub struct Broker {
    pub sessions: Sessions,
    client: Client,
    locks: Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, std::sync::Weak<tokio::sync::Mutex<()>>>,
        >,
    >,
}
impl Broker {
    /// Read the authoritative broker receipts without granting runtime callback
    /// authority or changing an uncertain operation's state.
    pub fn records(
        &self,
        principal: &Principal,
        session: &str,
        after: Option<&str>,
        limit: usize,
    ) -> Result<ToolPage> {
        self.sessions.get(principal, session)?;
        if !(1..=200).contains(&limit) {
            return Err(Fault::invalid("tool page limit must be 1..200"));
        }
        let prefix = key("tool", &[session]);
        if after.is_some_and(|cursor| !cursor.starts_with(&prefix)) {
            return Err(Fault::invalid("tool cursor belongs to another session"));
        }
        let mut records = self.sessions.store.scan::<ToolRecord>(&prefix)?;
        records.sort_by_key(|record| record_key(&record.call));
        records
            .retain(|record| after.is_none_or(|cursor| record_key(&record.call).as_str() > cursor));
        let more = records.len() > limit;
        records.truncate(limit);
        let next_cursor = more.then(|| record_key(&records.last().unwrap().call));
        Ok(ToolPage {
            records,
            next_cursor,
        })
    }
    fn call_lock(&self, call: &ToolCall) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(|e| e.into_inner());
        locks.retain(|_, lock| lock.strong_count() > 0);
        let entry = locks.entry(record_key(call)).or_default();
        if let Some(lock) = entry.upgrade() {
            return lock;
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        *entry = Arc::downgrade(&lock);
        lock
    }
    pub async fn reconcile_tool(
        &self,
        p: &Principal,
        call: &ToolCall,
        resolution: &Reconciliation,
    ) -> Result<ToolResult> {
        crate::http::role(p, "admin")?;
        let call_lock = self.call_lock(call);
        let _lock = call_lock.lock().await;
        self.sessions.store.transaction(|tx| {
            let target = record_key(call);
            let mut record: ToolRecord = tx
                .get(&target)?
                .ok_or_else(|| Fault::missing("tool record not found"))?;
            if record.call != *call {
                return Err(Fault::conflict("tool call differs from persisted request"));
            }
            crate::recovery::validate(p, call, resolution)?;
            if crate::recovery::already(tx, p, &target, resolution)? {
                return record
                    .result
                    .ok_or_else(|| Fault::conflict("resolved tool has no result"));
            }
            if !record.unknown && !(record.sent && record.result.is_none()) {
                return Err(Fault::conflict("tool result is already known"));
            }
            if matches!(record.destination, ToolDestination::Workspace { .. }) {
                return Err(Fault::invalid(
                    "reconcile workspace effects at akhexecd; its receipt remains authoritative",
                ));
            }
            let result = resolution.result.clone().ok_or_else(|| {
                Fault::invalid("tool reconciliation requires an inspected result")
            })?;
            if result.success != (resolution.status == TurnStatus::Completed)
                || resolution.output.is_some()
            {
                return Err(Fault::invalid(
                    "tool success must match its resolved status",
                ));
            }
            record.result = Some(result.clone());
            record.unknown = false;
            tx.put(&target, &record)?;
            crate::recovery::record(tx, p, &target, resolution)?;
            Ok(result)
        })
    }
    pub fn new(sessions: Sessions) -> Result<Self> {
        Ok(Self {
            client: Client::new(sessions.config.clone())?,
            sessions,
            locks: Default::default(),
        })
    }
    pub fn input(
        &self,
        principal: &Principal,
        call: &ToolCall,
        kind: &str,
        prompt: Value,
    ) -> Result<CallbackReply> {
        let session = self.sessions.validate_runtime_call(principal, call)?;
        if !matches!(kind, "native_approval" | "question") {
            return Err(Fault::invalid("invalid runtime input kind"));
        }
        if kind == "native_approval"
            && (session.runtime.execution_mode == ExecutionMode::Brokered
                || session.approval_policy == ApprovalPolicy::ReadOnly)
        {
            return Ok(CallbackReply {
                result: None,
                response: Some(json!({"approved":false})),
                pending: false,
            });
        }
        let input = self.sessions.request_input(principal, call, kind, prompt)?;
        Ok(CallbackReply {
            result: None,
            pending: input.response.is_none(),
            response: input.response,
        })
    }
    pub async fn tool(&self, principal: &Principal, call: ToolCall) -> Result<CallbackReply> {
        let call_lock = self.call_lock(&call);
        let _lock = call_lock.lock().await;
        let session = self.sessions.validate_runtime_call(principal, &call)?;
        let Some(tool) = session.tools.iter().find(|t| t.name == call.name) else {
            return Ok(done(ToolResult::denied(&call.name, None)));
        };
        let validator = jsonschema::validator_for(&tool.input_schema).map_err(|_| {
            Fault::invalid("configured tool schema is invalid or has external references")
        })?;
        if !validator.is_valid(&call.arguments) {
            return Ok(done(ToolResult {
                success: false,
                text: "Arguments do not match the advertised tool schema. Nothing ran.".into(),
                data: Some(json!({"code":"invalid_arguments","executed":false})),
                receipt: None,
            }));
        }
        // Serialize equivalent MCP effects within a session, including calls with
        // new model-generated IDs. A lost reply must not cause a duplicate write.
        let effect_lock = if matches!(tool.destination, ToolDestination::Mcp { .. }) {
            let mut effect = call.clone();
            effect.turn_id = "mcp-effect".into();
            effect.id = crate::recovery::fingerprint(
                &json!({"destination":tool.destination,"arguments":call.arguments}),
            )?;
            Some(self.call_lock(&effect))
        } else {
            None
        };
        let _effect_guard = match &effect_lock {
            Some(lock) => Some(lock.lock().await),
            None => None,
        };
        let k = record_key(&call);
        let mut record = match self.sessions.store.get::<ToolRecord>(&k)? {
            Some(old) => {
                if old.call != call {
                    return Err(Fault::conflict(
                        "native tool call ID reused with different arguments",
                    ));
                }
                if let Some(result) = old.result {
                    return Ok(done(result));
                }
                if old.unknown {
                    if matches!(old.destination, ToolDestination::Mcp { .. }) {
                        return Ok(done(mcp_uncertain_result(&old)));
                    }
                    return Err(Fault::new(
                        "unknown_outcome",
                        "tool delivery has an unknown outcome; it will not be repeated",
                    ));
                }
                old
            }
            None => {
                let record = ToolRecord {
                    operation_id: operation_id(&call),
                    call: call.clone(),
                    destination: tool.destination.clone(),
                    sent: false,
                    result: None,
                    unknown: false,
                };
                self.sessions.store.transaction(|tx| tx.put(&k, &record))?;
                record
            }
        };
        if matches!(tool.destination, ToolDestination::Mcp { .. }) {
            for previous in self
                .sessions
                .store
                .scan::<ToolRecord>(&key("tool", &[&session.id]))?
            {
                if previous.call != call
                    && previous.destination == tool.destination
                    && previous.call.arguments == call.arguments
                    && (previous.unknown || (previous.sent && previous.result.is_none()))
                {
                    return self.finish(k, record, mcp_uncertain_result(&previous));
                }
            }
        }
        // Validate built-in semantics independently of a mistaken read_only annotation.
        let writes = !tool.read_only
            || matches!(&tool.destination, ToolDestination::Workspace { operation } if !is_read_only_workspace_operation(operation));
        if writes && session.approval_policy == ApprovalPolicy::ReadOnly {
            return self.finish(k, record, ToolResult::denied(&call.name, None));
        }
        if writes && session.approval_policy == ApprovalPolicy::AskWrites {
            let mut approval = call.clone();
            approval.id = format!("approval-{}", record.operation_id);
            let pending = self.sessions.request_input(principal, &approval, "tool_approval", json!({"name":call.name,"arguments":call.arguments,"operation_id":record.operation_id}))?;
            match pending.response {
                None => {
                    return Ok(CallbackReply {
                        result: None,
                        response: None,
                        pending: true,
                    });
                }
                Some(response) if response["approved"] == true => {}
                Some(_) => return self.finish(k, record, ToolResult::denied(&call.name, None)),
            }
        }
        // An answer can arrive concurrently with cancellation. Recheck before dispatch.
        self.sessions.validate_runtime_call(principal, &call)?;
        let mcp_token = if let ToolDestination::Mcp { service, .. } = &tool.destination {
            Some(self.sessions.mcp_token(&session, service)?)
        } else {
            None
        };
        // Phase 53e: resolve the session's delegation grant before marking the
        // call sent, exactly like an MCP capability.
        let grant = if matches!(tool.destination, ToolDestination::Cognitive { .. }) {
            self.sessions.delegation_grant(&session.id)?
        } else {
            None
        };
        let was_sent = record.sent;
        record.sent = true;
        self.sessions.store.transaction(|tx| tx.put(&k, &record))?;
        let result: Result<ToolResult> = match &tool.destination {
            ToolDestination::Workspace { operation } => {
                let workspace = session.workspace.as_ref().ok_or_else(||Fault::conflict("workspace is not ready"))?;
                let operation = WorkspaceOperation { id: record.operation_id.clone(), session_id: session.id.clone(), turn_id: call.turn_id.clone(),
                    workspace_id: workspace.id.clone(), generation: workspace.generation, operation: operation.clone(), arguments: call.arguments.clone() };
                let receipt: OperationReceipt = self.client.in_sandbox(&workspace.sandbox_id)?.post(&workspace.executor, "/v1/operations", &operation).await?;
                if receipt.request != operation { return Err(Fault::conflict("executor returned a receipt for another request")); }
                if receipt.status == TurnStatus::Unknown { record.unknown = true; self.sessions.store.transaction(|tx|tx.put(&k, &record))?; return Err(Fault::new("unknown_outcome", "executor cannot confirm this operation")); }
                if operation.operation == "exec_start" && !receipt.status.terminal() {
                    Ok(ToolResult { success: true, text: "Process started; use the process handle to poll, send stdin or cancel.".into(),
                        data: Some(json!({"process_id":receipt.id,"status":receipt.status})), receipt: Some(receipt.id) })
                } else if let Some(result) = receipt.result { Ok(result) }
                else { return Ok(CallbackReply { result: None, response: None, pending: true }); }
            }
            ToolDestination::Cognitive { service, tool } => {
                self.client.post(service, "/v1/tools/call", &CognitiveToolCall { request_id: record.operation_id.clone(), session_id: session.id.clone(), akh: session.akh.clone(), tool: tool.clone(), arguments: call.arguments.clone(), grant }).await
            }
            ToolDestination::Mcp { service, tool } => {
                if was_sent { return self.mcp_uncertain(k, record, "previous response unavailable"); }
                let mut endpoint=self.sessions.config.endpoint(service)?.clone();
                if endpoint.url.contains("{sandbox}") {
                    let workspace=session.workspace.as_ref().ok_or_else(||Fault::missing("MCP sandbox is unbound"))?;
                    self.client.in_sandbox(&workspace.sandbox_id)?;
                    endpoint.url=endpoint.url.replace("{sandbox}",&workspace.sandbox_id);
                }
                match crate::mcp::call_with_token(&endpoint, mcp_token.expect("resolved MCP capability"), &record.operation_id, tool, call.arguments.clone()).await {
                    Ok(result) => Ok(result),
                    Err(error) => { return self.mcp_uncertain(k, record, &error.code); }
                }
            }
            ToolDestination::Client => {
                let mut client_call = call.clone(); client_call.id = format!("client-{}", record.operation_id);
                let pending = self.sessions.request_input(principal, &client_call, "client_tool", json!({"name":call.name,"arguments":call.arguments}))?;
                match pending.response {
                    Some(value) => serde_json::from_value(value).map_err(|_|Fault::invalid("client tool response must be a ToolResult")),
                    None => return Ok(CallbackReply { result: None, response: None, pending: true }),
                }
            }
        };
        match result {
            Ok(result) => self.finish(k, record, result),
            Err(error) if error.code == "unknown_outcome" => {
                record.unknown = true;
                self.sessions.store.transaction(|tx| tx.put(&k, &record))?;
                Err(error)
            }
            Err(error) if error.code == "transport" => Err(error),
            Err(error) => self.finish(
                k,
                record,
                ToolResult {
                    success: false,
                    text: error.to_string(),
                    data: Some(json!({"error":error})),
                    receipt: None,
                },
            ),
        }
    }
    fn mcp_uncertain(
        &self,
        k: String,
        mut record: ToolRecord,
        code: &str,
    ) -> Result<CallbackReply> {
        record.unknown = true;
        self.sessions.store.transaction(|tx| tx.put(&k, &record))?;
        tracing::warn!(event.name="mcp.operation.unknown", session.id=%record.call.session_id,
            turn.id=%record.call.turn_id, tool.name=%record.call.name,
            operation.id=%record.operation_id, error.code=code,
            "MCP response unavailable; operation retained for reconciliation without quarantining runtime");
        Ok(done(mcp_uncertain_result(&record)))
    }

    fn finish(
        &self,
        k: String,
        mut record: ToolRecord,
        result: ToolResult,
    ) -> Result<CallbackReply> {
        record.result = Some(result.clone());
        self.sessions.store.transaction(|tx| tx.put(&k, &record))?;
        Ok(done(result))
    }
    /// Provider text alone cannot close a turn while a process still has no terminal receipt.
    pub async fn settle(&self, mut job: RuntimeJob) -> Result<RuntimeJob> {
        if !job.status.terminal() {
            return Ok(job);
        }
        let records = self.sessions.store.scan::<ToolRecord>(&key(
            "tool",
            &[&job.request.session_id, &job.request.turn_id],
        ))?;
        for record in records {
            let call_lock = self.call_lock(&record.call);
            let _lock = call_lock.lock().await;
            let mut record: ToolRecord = self
                .sessions
                .store
                .get(&record_key(&record.call))?
                .ok_or_else(|| Fault::missing("tool record disappeared"))?;
            if (record.unknown || (record.sent && record.result.is_none()))
                && matches!(record.destination, ToolDestination::Workspace { .. })
            {
                let workspace = &job.request.workspace;
                let lookup = self
                    .client
                    .in_sandbox(&workspace.sandbox_id)?
                    .get::<OperationReceipt>(
                        &workspace.executor,
                        &format!("/v1/operations/{}", segment(&record.operation_id)),
                    )
                    .await;
                match RecoveryLookup::classify(&record.destination, lookup)? {
                    RecoveryLookup::Receipt(receipt) => {
                        if receipt.request.session_id != job.request.session_id
                            || receipt.request.turn_id != job.request.turn_id
                            || receipt.id != record.operation_id
                        {
                            return Err(Fault::conflict(
                                "executor recovery receipt belongs to another turn",
                            ));
                        }
                        if receipt.status.terminal() {
                            record.result = receipt.result;
                            record.unknown = false;
                            self.sessions
                                .store
                                .transaction(|tx| tx.put(&record_key(&record.call), &record))?;
                        }
                    }
                    RecoveryLookup::CollectedRead { operation } => {
                        // The read changed nothing, so an expired receipt
                        // cannot hide an effect: settle it as not applied and
                        // let the model re-issue the read if it still needs it.
                        tracing::info!(
                            event.name = "workspace.read_receipt.collected",
                            session.id = %record.call.session_id,
                            turn.id = %record.call.turn_id,
                            operation.id = %record.operation_id,
                            operation = %operation,
                            "read-only receipt was collected by akhexecd; settling the call as not applied"
                        );
                        record.result = Some(collected_read_result(&record, &operation));
                        record.unknown = false;
                        self.sessions
                            .store
                            .transaction(|tx| tx.put(&record_key(&record.call), &record))?;
                    }
                }
            }
            if matches!(record.destination, ToolDestination::Mcp { .. }) {
                // The remote outcome remains in the broker ledger. It is not
                // evidence of a live local process, and cannot quarantine an Akh.
                continue;
            }
            if record.unknown || (record.sent && record.result.is_none()) {
                job.status = TurnStatus::Unknown;
                job.error = Some(Fault::new("unknown_outcome", "tool outcome is unresolved"));
                break;
            }
            if matches!(&record.destination, ToolDestination::Workspace { operation } if operation == "exec_start")
                && record.sent
            {
                let workspace = &job.request.workspace;
                let client = self.client.in_sandbox(&workspace.sandbox_id)?;
                let mut receipt: OperationReceipt = client
                    .get(
                        &workspace.executor,
                        &format!("/v1/operations/{}", segment(&record.operation_id)),
                    )
                    .await?;
                if !receipt.status.terminal() && job.status == TurnStatus::Cancelled {
                    let request = WorkspaceOperation {
                        id: format!("cancel-{}", record.operation_id),
                        session_id: job.request.session_id.clone(),
                        turn_id: job.request.turn_id.clone(),
                        workspace_id: workspace.id.clone(),
                        generation: workspace.generation,
                        operation: "exec_cancel".into(),
                        arguments: json!({"process_id":record.operation_id}),
                    };
                    let _: OperationReceipt = client
                        .post(&workspace.executor, "/v1/operations", &request)
                        .await?;
                    receipt = client
                        .get(
                            &workspace.executor,
                            &format!("/v1/operations/{}", segment(&record.operation_id)),
                        )
                        .await?;
                }
                if receipt.status == TurnStatus::Unknown {
                    job.status = TurnStatus::Unknown;
                    job.error = Some(Fault::new(
                        "unknown_outcome",
                        "process has no terminal receipt",
                    ));
                    break;
                }
                if !receipt.status.terminal() {
                    job.status = TurnStatus::Running;
                    job.output = None;
                }
            }
        }
        Ok(job)
    }
}
/// What a broker recovery lookup of a workspace receipt found.
enum RecoveryLookup {
    /// akhexecd still holds the receipt; it stays authoritative.
    Receipt(Box<OperationReceipt>),
    /// akhexecd garbage-collected the terminal receipt of a read-only verb
    /// (see `COLLECTABLE_READ_OPERATIONS`). Reads have no effect, so the call
    /// is safe to treat as not applied.
    CollectedRead { operation: String },
}

impl RecoveryLookup {
    /// Classify a `GET /v1/operations/{id}` result. A `not_found` is safe only
    /// for a collectable read-only verb; for every other operation (mutations,
    /// `exec_start`, `git_snapshot`) it stays a hard failure, because the
    /// receipt is the only evidence of what ran.
    fn classify(
        destination: &ToolDestination,
        lookup: Result<OperationReceipt>,
    ) -> Result<Self> {
        match lookup {
            Ok(receipt) => Ok(Self::Receipt(Box::new(receipt))),
            Err(error) if error.code == "not_found" => match destination {
                ToolDestination::Workspace { operation }
                    if COLLECTABLE_READ_OPERATIONS.contains(&operation.as_str()) =>
                {
                    Ok(Self::CollectedRead {
                        operation: operation.clone(),
                    })
                }
                _ => Err(error),
            },
            Err(error) => Err(error),
        }
    }
}

fn collected_read_result(record: &ToolRecord, operation: &str) -> ToolResult {
    ToolResult {
        success: false,
        text: format!(
            "The receipt for read-only operation {} ({operation}) has expired, so its result is unavailable. Nothing was changed; re-issue the {operation} if you still need it.",
            record.operation_id
        ),
        data: Some(json!({
            "code":"read_receipt_collected",
            "outcome":"not_applied",
            "operation":operation,
            "retryable":true,
            "operation_id":record.operation_id,
        })),
        receipt: Some(record.operation_id.clone()),
    }
}

fn done(result: ToolResult) -> CallbackReply {
    CallbackReply {
        result: Some(result),
        response: None,
        pending: false,
    }
}

fn mcp_uncertain_result(record: &ToolRecord) -> ToolResult {
    ToolResult {
        success: false,
        text: format!(
            "MCP operation {} has an unknown outcome because its response was unavailable. It may have executed remotely. Do not claim success or repeat this operation; inspect the remote service before reconciling it. Other independent work may continue.",
            record.operation_id
        ),
        data: Some(
            json!({"code":"mcp_outcome_unknown","outcome":"unknown","retryable":false,"operation_id":record.operation_id}),
        ),
        receipt: Some(record.operation_id.clone()),
    }
}
