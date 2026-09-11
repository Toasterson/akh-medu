//! Session-scoped tool admission and durable downstream execution evidence.
use crate::{
    config::Principal,
    http::{Client, segment},
    runtime::CallbackReply,
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
fn record_key(call: &ToolCall) -> String {
    key("tool", &[&call.session_id, &call.turn_id, &call.id])
}
pub fn operation_id(call: &ToolCall) -> String {
    crate::workspace::digest(record_key(call).as_bytes())
}

#[derive(Clone)]
pub struct Broker {
    pub sessions: Sessions,
    client: Client,
    lock: Arc<tokio::sync::Mutex<()>>,
}
impl Broker {
    pub fn new(sessions: Sessions) -> Result<Self> {
        Ok(Self {
            client: Client::new(sessions.config.clone())?,
            sessions,
            lock: Default::default(),
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
        let _lock = self.lock.lock().await;
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
        // Validate built-in semantics independently of a mistaken read_only annotation.
        let writes = !tool.read_only
            || matches!(&tool.destination, ToolDestination::Workspace { operation } if !matches!(operation.as_str(), "read"|"list"|"search"|"exec_poll"));
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
                self.client.post(service, "/v1/tools/call", &json!({"request_id":record.operation_id,"session_id":session.id,"akh":session.akh,"tool":tool,"arguments":call.arguments})).await
            }
            ToolDestination::Mcp { service, tool } => {
                if was_sent { record.unknown = true; self.sessions.store.transaction(|tx|tx.put(&k, &record))?; return Err(Fault::new("unknown_outcome", "MCP has no standard idempotency guarantee; inspect the server before reconciling")); }
                let reply: Result<Value> = self.client.post(service, "", &json!({"jsonrpc":"2.0","id":record.operation_id,"method":"tools/call","params":{"name":tool,"arguments":call.arguments}})).await;
                match reply {
                    Ok(reply) if reply.get("error").is_none() => Ok(ToolResult { success: reply["result"]["isError"] != true, text: reply["result"]["content"].to_string(), data: Some(reply["result"].clone()), receipt: None }),
                    Ok(_) => Err(Fault::new("tool_error", "MCP server rejected the tool call")),
                    Err(error) => { record.unknown = true; self.sessions.store.transaction(|tx|tx.put(&k, &record))?; Err(error) }
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
fn done(result: ToolResult) -> CallbackReply {
    CallbackReply {
        result: Some(result),
        response: None,
        pending: false,
    }
}
