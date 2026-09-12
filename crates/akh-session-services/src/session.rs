use crate::{
    config::{Config, Principal},
    store::{Store, Transaction, key},
    types::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[derive(Clone)]
pub struct Sessions {
    pub store: Store,
    pub config: Arc<Config>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CommandRecord {
    request: Value,
    result: String,
}

pub use akh_session_types::Admission;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AutomationRun {
    request: InvokeAutomation,
    admission: Admission,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RevokedAutomation {
    principal: String,
    request: RevokeAutomation,
    at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct RecordedInputDecision {
    principal: String,
    decision: InputDecision,
}

fn encode(value: &impl Serialize) -> Result<Value> {
    serde_json::to_value(value).map_err(|e| Fault::invalid(e.to_string()))
}

fn record_key(id: &str) -> String {
    key("session", &[id])
}

fn load(tx: &Transaction<'_>, id: &str) -> Result<SessionRecord> {
    tx.get(&record_key(id))?
        .ok_or_else(|| Fault::missing("session not found"))
}

/// Checked inside the same transaction as each client mutation. A transport's
/// earlier read cannot authorize a write after another attachment takes control.
fn check_controller(session: &SessionRecord, attachment: Option<&str>) -> Result<()> {
    if session.controller.as_ref().is_some_and(|lease| {
        lease.expires_at > now_ms() && Some(lease.attachment_id.as_str()) != attachment
    }) {
        return Err(Fault::conflict(
            "another client holds this session's controller lease",
        ));
    }
    Ok(())
}

fn authorize(session: &SessionRecord, principal: &Principal) -> Result<()> {
    if session.principal != principal.id && !principal.has_role("admin") {
        return Err(Fault::new(
            "forbidden",
            "session belongs to another principal",
        ));
    }
    Ok(())
}

/// Validate without writing. Only semantic failures from this phase may become
/// negative receipts; a storage failure must roll the whole transaction back.
fn prepare_response(
    tx: &Transaction<'_>,
    session: &SessionRecord,
    turn_id: &str,
    id: &str,
    generation: Option<u64>,
    response: &Value,
) -> Result<PendingInput> {
    let pending: PendingInput = tx
        .get(&key("pending", &[&session.id, turn_id, id]))?
        .ok_or_else(|| Fault::missing("pending input not found"))?;
    if generation.is_some_and(|expected| expected != pending.generation) {
        return Err(Fault::conflict("pending input generation changed"));
    }
    if let Some(old) = &pending.response {
        if old != response {
            return Err(Fault::conflict("input already answered differently"));
        }
        return Ok(pending);
    }
    if pending.kind == "question" && pending.prompt["format"] == questions::QuestionForm::FORMAT {
        let form: questions::QuestionForm = serde_json::from_value(pending.prompt.clone())
            .map_err(|_| Fault::invalid("invalid stored question form"))?;
        let answers: questions::QuestionAnswers = serde_json::from_value(response.clone())
            .map_err(|_| Fault::invalid("invalid question answers"))?;
        form.validate_answers(&answers).map_err(Fault::invalid)?;
    }
    let turn = session
        .turns
        .iter()
        .find(|turn| turn.id == turn_id)
        .ok_or_else(|| Fault::missing("turn not found"))?;
    if pending.generation != turn.generation
        || turn.status.terminal()
        || turn.status == TurnStatus::Unknown
        || turn.cancel_requested
    {
        return Err(Fault::conflict(
            "pending input no longer belongs to an active turn",
        ));
    }
    Ok(pending)
}

fn apply_response(
    tx: &mut Transaction<'_>,
    session: &mut SessionRecord,
    mut pending: PendingInput,
    response: Value,
) -> Result<PendingInput> {
    if pending.response.is_some() {
        return Ok(pending);
    }
    let turn = session
        .turns
        .iter_mut()
        .find(|turn| turn.id == pending.turn_id)
        .ok_or_else(|| Fault::missing("turn not found"))?;
    pending.response = Some(response);
    turn.status = TurnStatus::Running;
    tx.put(
        &key("pending", &[&session.id, &pending.turn_id, &pending.id]),
        &pending,
    )?;
    append_event(
        tx,
        session,
        Some(&pending.turn_id),
        "input_answered",
        json!({"id":pending.id}),
    )?;
    Ok(pending)
}

pub fn append_event(
    tx: &mut Transaction<'_>,
    session: &mut SessionRecord,
    turn_id: Option<&str>,
    kind: &str,
    data: Value,
) -> Result<SessionEvent> {
    let event = SessionEvent {
        schema_version: SCHEMA_VERSION,
        session_id: session.id.clone(),
        sequence: session.next_sequence,
        turn_id: turn_id.map(str::to_owned),
        kind: kind.into(),
        data,
        at: now_ms(),
    };
    session.next_sequence = session
        .next_sequence
        .checked_add(1)
        .ok_or_else(|| Fault::conflict("event sequence exhausted"))?;
    tx.put(
        &key("event", &[&session.id, &format!("{:020}", event.sequence)]),
        &event,
    )?;
    tx.put(&record_key(&session.id), session)?;
    Ok(event)
}

impl Sessions {
    /// Irreversibly stop accepting new inputs and request cancellation of all
    /// outstanding turns. Unknown outcomes remain unknown until audited recovery.
    /// This receipt proves intent, not provider termination or sandbox cleanup.
    pub fn retire(
        &self,
        p: &Principal,
        id: &str,
        attachment: Option<&str>,
        request: &RetireSession,
    ) -> Result<SessionRetirement> {
        validate_key(&request.request_id)?;
        if request.reason.trim().is_empty() || request.reason.len() > 4096 {
            return Err(Fault::invalid("retirement requires a bounded audit reason"));
        }
        self.store.transaction(|tx| {
            let mut session = load(tx, id)?;
            authorize(&session, p)?;
            if let Some(prior) = &session.retirement {
                if prior.request != *request || prior.principal != p.id {
                    return Err(Fault::conflict(
                        "session retirement already belongs to another intent",
                    ));
                }
                return Ok(prior.clone());
            }
            check_controller(&session, attachment)?;
            if session.closed_at.is_some() {
                return Err(Fault::conflict("session is already archived"));
            }
            let retirement = SessionRetirement {
                request: request.clone(),
                principal: p.id.clone(),
                at: now_ms(),
            };
            session.retirement = Some(retirement.clone());
            append_event(
                tx,
                &mut session,
                None,
                "session_retiring",
                encode(&retirement)?,
            )?;
            for index in 0..session.turns.len() {
                let turn = &mut session.turns[index];
                if turn.status.terminal() || turn.cancel_requested {
                    continue;
                }
                turn.cancel_requested = true;
                if turn.status == TurnStatus::Queued {
                    turn.status = TurnStatus::Cancelled;
                }
                let id = turn.id.clone();
                let status = turn.status;
                append_event(
                    tx,
                    &mut session,
                    Some(&id),
                    "cancel_requested",
                    json!({"status":status}),
                )?;
            }
            Ok(retirement)
        })
    }

    pub fn archive(&self, p: &Principal, id: &str) -> Result<SessionRecord> {
        self.archive_controlled(p, id, None)
    }

    pub fn archive_controlled(
        &self,
        p: &Principal,
        id: &str,
        attachment: Option<&str>,
    ) -> Result<SessionRecord> {
        self.store.transaction(|tx| {
            let mut session = load(tx, id)?;
            authorize(&session, p)?;
            check_controller(&session, attachment)?;
            if session.closed_at.is_some() {
                return Ok(session);
            }
            if session.turns.iter().any(|turn| !turn.status.terminal()) {
                return Err(Fault::conflict(
                    "cancel or reconcile outstanding turns before archiving",
                ));
            }
            session.closed_at = Some(now_ms());
            session.controller = None;
            append_event(tx, &mut session, None, "session_archived", json!({}))?;
            Ok(session)
        })
    }
    pub fn new(store: Store, config: Arc<Config>) -> Self {
        Self { store, config }
    }

    fn create_in(
        &self,
        tx: &mut Transaction<'_>,
        principal: &Principal,
        request: &NewSession,
    ) -> Result<SessionRecord> {
        validate_key(&request.request_id)?;
        if !principal.can_use_akh(&request.akh) {
            return Err(Fault::new(
                "forbidden",
                "Akh is not granted to this principal",
            ));
        }
        let profile = self
            .config
            .akhs
            .get(&request.akh)
            .ok_or_else(|| Fault::missing("Akh profile not found"))?;
        if !profile
            .workspace_profiles
            .contains(&request.workspace_profile)
        {
            return Err(Fault::new(
                "forbidden",
                "workspace profile is not granted to this Akh",
            ));
        }
        let dedupe = key("create", &[&principal.id, &request.request_id]);
        let request_value = encode(request)?;
        if let Some(old) = tx.get::<CommandRecord>(&dedupe)? {
            if old.request != request_value {
                return Err(Fault::conflict(
                    "request_id already used with different session parameters",
                ));
            }
            return load(tx, &old.result);
        }
        let conversation_key = request
            .conversation_key
            .as_ref()
            .map(|k| key("conversation", &[&principal.id, &request.akh, k]));
        if let Some(k) = &request.conversation_key {
            validate_key(k)?;
        }
        if let Some(index) = &conversation_key {
            if let Some(id) = tx.get::<String>(index)? {
                let session = load(tx, &id)?;
                if session.workspace_profile != request.workspace_profile {
                    return Err(Fault::conflict(
                        "conversation already has a different workspace profile",
                    ));
                }
                tx.put(
                    &dedupe,
                    &CommandRecord {
                        request: request_value,
                        result: id,
                    },
                )?;
                return Ok(session);
            }
        }
        let mut session = SessionRecord {
            repository_bootstrap: crate::session_bootstrap::initial(profile)?,
            required_mcp_credentials: self.required_mcp_credentials(&profile.tools)?,
            retirement: None,
            closed_at: None,
            core_service: profile.core_service.clone(),
            schema_version: SCHEMA_VERSION,
            id: new_id(),
            principal: principal.id.clone(),
            akh: request.akh.clone(),
            conversation_key: request.conversation_key.clone(),
            workspace_profile: request.workspace_profile.clone(),
            workspace: None,
            runtime: profile.runtime.clone(),
            instructions: profile.instructions.clone(),
            tools: profile.tools.clone(),
            approval_policy: profile.approval_policy,
            generation: 0,
            next_sequence: 1,
            turns: vec![],
            provider_session: None,
            controller: None,
            created_at: now_ms(),
        };
        tx.put(
            &dedupe,
            &CommandRecord {
                request: request_value,
                result: session.id.clone(),
            },
        )?;
        if let Some(index) = &conversation_key {
            tx.put(index, &session.id)?;
        }
        append_event(
            tx,
            &mut session,
            None,
            "session_created",
            json!({"akh": request.akh}),
        )?;
        Ok(session)
    }

    pub fn create(&self, principal: &Principal, request: &NewSession) -> Result<SessionRecord> {
        self.store
            .transaction(|tx| self.create_in(tx, principal, request))
    }

    pub fn get(&self, principal: &Principal, id: &str) -> Result<SessionRecord> {
        let session: SessionRecord = self
            .store
            .get(&record_key(id))?
            .ok_or_else(|| Fault::missing("session not found"))?;
        authorize(&session, principal)?;
        Ok(session)
    }

    pub fn list(&self, principal: &Principal) -> Result<Vec<SessionRecord>> {
        Ok(self
            .store
            .scan::<SessionRecord>("session/")?
            .into_iter()
            .filter(|s| authorize(s, principal).is_ok())
            .collect())
    }

    pub fn events(&self, principal: &Principal, id: &str, after: u64) -> Result<Vec<SessionEvent>> {
        self.get(principal, id)?;
        Ok(self
            .store
            .scan::<SessionEvent>(&key("event", &[id]))?
            .into_iter()
            .filter(|e| e.sequence > after)
            .take(1000)
            .collect())
    }

    fn submit_in(
        tx: &mut Transaction<'_>,
        principal: &Principal,
        id: &str,
        request_id: &str,
        input: TurnInput,
    ) -> Result<TurnRecord> {
        validate_key(request_id)?;
        if input.text.trim().is_empty() || input.text.len() > 1024 * 1024 {
            return Err(Fault::invalid(
                "input text must be non-empty and at most 1 MiB",
            ));
        }
        if !input.artifacts.is_empty() {
            return Err(Fault::new(
                "unsupported",
                "artifact inputs are not yet supported end to end; use workspace tools",
            ));
        }
        let mut session = load(tx, id)?;
        authorize(&session, principal)?;
        let dedupe = key("input", &[id, &principal.id, request_id]);
        let value = encode(&input)?;
        if let Some(old) = tx.get::<CommandRecord>(&dedupe)? {
            if old.request != value {
                return Err(Fault::conflict(
                    "input request_id reused with different content",
                ));
            }
            return session
                .turns
                .iter()
                .find(|t| t.id == old.result)
                .cloned()
                .ok_or_else(|| Fault::new("storage_error", "admitted turn is missing"));
        }
        if session.closed_at.is_some() || session.retirement.is_some() {
            return Err(Fault::conflict(
                "session is retiring or archived; create an explicit new session",
            ));
        }
        if session
            .turns
            .iter()
            .filter(|t| !t.status.terminal())
            .count()
            >= 128
        {
            return Err(Fault::new("busy", "session input queue is full"));
        }
        let id = new_id();
        let turn = TurnRecord {
            id: id.clone(),
            request_id: request_id.into(),
            input,
            status: TurnStatus::Queued,
            generation: 0,
            runtime_job_id: id.clone(),
            cancel_requested: false,
            output: None,
            error: None,
            created_at: now_ms(),
        };
        session.turns.push(turn.clone());
        tx.put(
            &dedupe,
            &CommandRecord {
                request: value,
                result: id.clone(),
            },
        )?;
        append_event(
            tx,
            &mut session,
            Some(&id),
            "input_accepted",
            encode(&turn.input)?,
        )?;
        Ok(turn)
    }

    pub fn submit(
        &self,
        principal: &Principal,
        id: &str,
        request: &SubmitInput,
    ) -> Result<TurnRecord> {
        self.submit_controlled(principal, id, None, request)
    }

    pub fn submit_controlled(
        &self,
        principal: &Principal,
        id: &str,
        attachment: Option<&str>,
        request: &SubmitInput,
    ) -> Result<TurnRecord> {
        self.store.transaction(|tx| {
            let session = load(tx, id)?;
            authorize(&session, principal)?;
            check_controller(&session, attachment)?;
            Self::submit_in(
                tx,
                principal,
                id,
                &request.request_id,
                TurnInput {
                    origin: InputOrigin::Human {
                        principal: principal.id.clone(),
                    },
                    text: request.text.clone(),
                    data: None,
                    artifacts: request.artifacts.clone(),
                },
            )
        })
    }

    pub fn bind_workspace(&self, id: &str, binding: WorkspaceBinding) -> Result<()> {
        self.bind_prepared_workspace(id, binding, None)
    }

    pub fn bind_prepared_workspace(
        &self,
        id: &str,
        binding: WorkspaceBinding,
        result: Option<bootstrap::RepositoryBootstrapResult>,
    ) -> Result<()> {
        self.store.transaction(|tx| {
            let mut session = load(tx, id)?;
            crate::session_bootstrap::validate_result(&session, &binding, result.as_ref())?;
            if let Some(old) = &session.workspace {
                if old != &binding {
                    return Err(Fault::conflict(
                        "session is already bound to another workspace/generation",
                    ));
                }
                return Ok(());
            }
            if session.closed_at.is_some() {
                return Err(Fault::conflict(
                    "cannot bind a new workspace to an archived session",
                ));
            }
            session.workspace = Some(binding.clone());
            if let Some(result) = result {
                session
                    .repository_bootstrap
                    .as_mut()
                    .expect("validated bootstrap slot")
                    .result = Some(result.clone());
                append_event(
                    tx,
                    &mut session,
                    None,
                    "repository_bootstrap_completed",
                    encode(&result)?,
                )?;
            }
            append_event(tx, &mut session, None, "workspace_ready", encode(&binding)?)?;
            Ok(())
        })
    }

    /// Persist the dispatch claim before any request is sent to a runtime.
    pub fn next_job(&self, id: &str) -> Result<Option<RuntimeJobRequest>> {
        self.store.transaction(|tx| {
            let mut session = load(tx, id)?;
            let Some(workspace) = session.workspace.clone() else {
                return Ok(None);
            };
            let index = session.turns.iter().position(|t| !t.status.terminal());
            let Some(index) = index else {
                return Ok(None);
            };
            // Return the exact dispatch bytes even after a provider continuation is bound.
            if session.turns[index].status == TurnStatus::Unknown {
                return Ok(None);
            }
            let dispatch_key = key("dispatch", &[&session.turns[index].runtime_job_id]);
            if let Some(request) = tx.get::<RuntimeJobRequest>(&dispatch_key)? {
                return Ok(Some(request));
            }
            if session.turns[index].status == TurnStatus::Unknown {
                return Ok(None);
            }
            if session.turns[index].status == TurnStatus::Queued {
                self.check_mcp_credentials(tx, &session)?;
                self.check_repository_bootstrap(&session, true)?;
                session.generation = session
                    .generation
                    .checked_add(1)
                    .ok_or_else(|| Fault::conflict("generation exhausted"))?;
                session.turns[index].generation = session.generation;
                session.turns[index].status = TurnStatus::Dispatching;
                let turn_id = session.turns[index].id.clone();
                let generation = session.generation;
                append_event(
                    tx,
                    &mut session,
                    Some(&turn_id),
                    "turn_dispatching",
                    json!({"generation":generation}),
                )?;
            }
            let turn = &session.turns[index];
            let request = RuntimeJobRequest {
                id: turn.runtime_job_id.clone(),
                session_id: session.id.clone(),
                turn_id: turn.id.clone(),
                generation: turn.generation,
                binding: session.runtime.clone(),
                workspace,
                provider_session: session.provider_session.clone(),
                input: turn.input.clone(),
                instructions: tx
                    .get::<ContextResponse>(&key("turn_context", &[&turn.id]))?
                    .map(|context| format!("{}\n\n{}", context.instructions, session.instructions))
                    .unwrap_or_else(|| session.instructions.clone()),
                tools: session.tools.clone(),
                approval_policy: session.approval_policy,
            };
            tx.put(&dispatch_key, &request)?;
            Ok(Some(request))
        })
    }

    pub fn apply_job(&self, job: &RuntimeJob) -> Result<()> {
        self.store.transaction(|tx| {
            if tx
                .get::<RuntimeJobRequest>(&key("dispatch", &[&job.request.id]))?
                .as_ref()
                != Some(&job.request)
            {
                return Err(Fault::conflict(
                    "runtime response differs from the exact persisted dispatch",
                ));
            }
            let mut session = load(tx, &job.request.session_id)?;
            let index = session
                .turns
                .iter()
                .position(|t| t.id == job.request.turn_id && t.runtime_job_id == job.request.id)
                .ok_or_else(|| Fault::conflict("unknown runtime turn"))?;
            let turn = &session.turns[index];
            if turn.generation != job.request.generation
                || session.generation != job.request.generation
                || session.runtime != job.request.binding
            {
                return Err(Fault::conflict("stale runtime generation or binding"));
            }
            if turn.status.terminal() {
                return Ok(());
            }
            if let Some(provider_id) = &job.provider_session {
                if session
                    .provider_session
                    .as_ref()
                    .is_some_and(|id| id != provider_id)
                {
                    return Err(Fault::conflict("runtime replaced the provider session"));
                }
                session.provider_session = Some(provider_id.clone());
            }
            let cursor_key = key("runtime_cursor", &[&job.request.id]);
            let cursor = tx.get::<usize>(&cursor_key)?.unwrap_or(0);
            if cursor > job.events.len() {
                return Err(Fault::conflict("runtime event history moved backwards"));
            }
            let history_key = key("runtime_history", &[&job.request.id]);
            if let Some(expected) = tx.get::<String>(&history_key)? {
                if expected != crate::recovery::fingerprint(&job.events[..cursor])? {
                    return Err(Fault::conflict("runtime rewrote an already accepted event"));
                }
            }
            for event in &job.events[cursor..] {
                append_event(
                    tx,
                    &mut session,
                    Some(&job.request.turn_id),
                    "runtime",
                    encode(event)?,
                )?;
            }
            tx.put(&cursor_key, &job.events.len())?;
            tx.put(&history_key, &crate::recovery::fingerprint(&job.events)?)?;
            let awaiting = tx
                .scan::<PendingInput>(&key("pending", &[&session.id]))?
                .iter()
                .any(|p| p.generation == job.request.generation && p.response.is_none());
            let status = if awaiting && job.status == TurnStatus::Running {
                TurnStatus::AwaitingInput
            } else {
                job.status
            };
            let changed = session.turns[index].status != status;
            session.turns[index].status = status;
            session.turns[index].output = job.output.clone();
            session.turns[index].error = job.error.clone();
            if changed {
                append_event(
                    tx,
                    &mut session,
                    Some(&job.request.turn_id),
                    "turn_status",
                    json!({"status":status,"output":job.output,"error":job.error}),
                )?;
            } else {
                tx.put(&record_key(&session.id), &session)?;
            }
            Ok(())
        })
    }

    pub fn cancel(&self, principal: &Principal, id: &str, turn_id: &str) -> Result<TurnRecord> {
        self.cancel_controlled(principal, id, turn_id, None)
    }

    pub fn cancel_controlled(
        &self,
        principal: &Principal,
        id: &str,
        turn_id: &str,
        attachment: Option<&str>,
    ) -> Result<TurnRecord> {
        self.store.transaction(|tx| {
            let mut session = load(tx, id)?;
            authorize(&session, principal)?;
            check_controller(&session, attachment)?;
            let turn = session
                .turns
                .iter_mut()
                .find(|t| t.id == turn_id)
                .ok_or_else(|| Fault::missing("turn not found"))?;
            if turn.status.terminal() || turn.cancel_requested {
                return Ok(turn.clone());
            }
            turn.cancel_requested = true;
            if turn.status == TurnStatus::Queued {
                turn.status = TurnStatus::Cancelled;
            }
            let result = turn.clone();
            append_event(
                tx,
                &mut session,
                Some(turn_id),
                "cancel_requested",
                json!({"status":result.status}),
            )?;
            Ok(result)
        })
    }

    pub fn attach(
        &self,
        principal: &Principal,
        id: &str,
        attachment_id: &str,
        control: bool,
    ) -> Result<SessionRecord> {
        validate_key(attachment_id)?;
        self.store.transaction(|tx| {
            let mut session = load(tx, id)?;
            authorize(&session, principal)?;
            if control {
                if session
                    .controller
                    .as_ref()
                    .is_some_and(|l| l.attachment_id != attachment_id && l.expires_at > now_ms())
                {
                    return Err(Fault::conflict("another attachment holds control"));
                }
                session.controller = Some(ControllerLease {
                    attachment_id: attachment_id.into(),
                    expires_at: now_ms() + 60_000,
                });
                tx.put(&record_key(id), &session)?;
            }
            Ok(session)
        })
    }

    pub fn detach(&self, principal: &Principal, id: &str, attachment_id: &str) -> Result<()> {
        self.store.transaction(|tx| {
            let mut session = load(tx, id)?;
            authorize(&session, principal)?;
            if session
                .controller
                .as_ref()
                .is_some_and(|l| l.attachment_id == attachment_id)
            {
                session.controller = None;
                tx.put(&record_key(id), &session)?;
            }
            Ok(())
        })
    }

    pub fn validate_runtime_call(
        &self,
        principal: &Principal,
        call: &ToolCall,
    ) -> Result<SessionRecord> {
        let session: SessionRecord = self
            .store
            .get(&record_key(&call.session_id))?
            .ok_or_else(|| Fault::missing("session not found"))?;
        if principal.id != session.runtime.runtime || !principal.has_role("runtime") {
            return Err(Fault::new("forbidden", "runtime does not own this session"));
        }
        let active = session
            .turns
            .iter()
            .find(|t| t.id == call.turn_id)
            .ok_or_else(|| Fault::missing("turn not found"))?;
        if session.generation != call.generation
            || active.generation != call.generation
            || active.status.terminal()
            || active.status == TurnStatus::Unknown
            || active.cancel_requested
        {
            return Err(Fault::conflict("runtime call is stale or turn cancelled"));
        }
        validate_key(&call.id)?;
        Ok(session)
    }

    pub fn request_input(
        &self,
        principal: &Principal,
        call: &ToolCall,
        kind: &str,
        prompt: Value,
    ) -> Result<PendingInput> {
        self.validate_runtime_call(principal, call)?;
        if kind == "question" && prompt["format"] == questions::QuestionForm::FORMAT {
            let form: questions::QuestionForm = serde_json::from_value(prompt.clone())
                .map_err(|_| Fault::invalid("invalid question form"))?;
            form.validate().map_err(Fault::invalid)?;
        }
        self.store.transaction(|tx| {
            let mut session = load(tx, &call.session_id)?;
            let turn = session
                .turns
                .iter_mut()
                .find(|t| t.id == call.turn_id)
                .ok_or_else(|| Fault::missing("turn not found"))?;
            if turn.cancel_requested
                || turn.status.terminal()
                || turn.status == TurnStatus::Unknown
                || turn.generation != call.generation
                || session.generation != call.generation
            {
                return Err(Fault::conflict("turn is no longer accepting input"));
            }
            let id = key("pending", &[&call.session_id, &call.turn_id, &call.id]);
            if let Some(old) = tx.get::<PendingInput>(&id)? {
                if old.prompt != prompt || old.kind != kind {
                    return Err(Fault::conflict("input ID was reused"));
                }
                return Ok(old);
            }
            let pending = PendingInput {
                id: call.id.clone(),
                session_id: call.session_id.clone(),
                turn_id: call.turn_id.clone(),
                generation: call.generation,
                kind: kind.into(),
                prompt,
                response: None,
            };
            turn.status = TurnStatus::AwaitingInput;
            tx.put(&id, &pending)?;
            append_event(
                tx,
                &mut session,
                Some(&call.turn_id),
                "input_required",
                encode(&pending)?,
            )?;
            Ok(pending)
        })
    }

    pub fn respond(
        &self,
        principal: &Principal,
        session_id: &str,
        turn_id: &str,
        id: &str,
        response: Value,
    ) -> Result<PendingInput> {
        self.respond_if_generation(principal, session_id, turn_id, id, None, response)
    }

    /// Conditional input acknowledgement for clients that persist the observed
    /// generation. Check it before either recording or replaying a decision.
    pub fn respond_if_generation(
        &self,
        principal: &Principal,
        session_id: &str,
        turn_id: &str,
        id: &str,
        generation: Option<u64>,
        response: Value,
    ) -> Result<PendingInput> {
        self.respond_controlled(
            principal, session_id, turn_id, id, generation, None, response,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn respond_controlled(
        &self,
        principal: &Principal,
        session_id: &str,
        turn_id: &str,
        id: &str,
        generation: Option<u64>,
        attachment: Option<&str>,
        response: Value,
    ) -> Result<PendingInput> {
        self.store.transaction(|tx| {
            let mut session = load(tx, session_id)?;
            authorize(&session, principal)?;
            check_controller(&session, attachment)?;
            let pending = prepare_response(tx, &session, turn_id, id, generation, &response)?;
            apply_response(tx, &mut session, pending, response)
        })
    }

    /// Atomically persist positive or negative input decisions. Replaying one
    /// reads its original result, even if the target later appears or advances.
    /// Lease conflicts and storage/auth failures remain retriable HTTP errors.
    pub fn decide_input(
        &self,
        principal: &Principal,
        session_id: &str,
        attachment: Option<&str>,
        request: &InputDecisionRequest,
    ) -> Result<InputDecision> {
        validate_key(&request.request_id)?;
        validate_key(&request.turn_id)?;
        validate_key(&request.input_id)?;
        if request.generation == 0
            || serde_json::to_vec(&request.response)
                .map_err(|_| Fault::invalid("input response encoding"))?
                .len()
                > 64 * 1024
        {
            return Err(Fault::invalid(
                "input decision needs positive generation and response at most 64 KiB",
            ));
        }
        self.store.transaction(|tx| {
            let mut session = load(tx, session_id)?;
            authorize(&session, principal)?;
            let target=key("input_decision", &[session_id, &request.request_id]);
            if let Some(old)=tx.get::<RecordedInputDecision>(&target)? {
                if old.principal != principal.id || old.decision.request != *request {
                    return Err(Fault::conflict("input decision identity reused with different intent or principal"));
                }
                return Ok(old.decision);
            }
            check_controller(&session, attachment)?;
            let rejection=match prepare_response(tx, &session, &request.turn_id,
                &request.input_id, Some(request.generation), &request.response) {
                Ok(pending) => {
                    apply_response(tx, &mut session, pending, request.response.clone())?;
                    None
                },
                Err(error) if matches!(error.code.as_str(), "conflict"|"not_found") => Some(error),
                Err(error) => return Err(error),
            };
            let decision=InputDecision {schema_version:SCHEMA_VERSION,session_id:session_id.into(),
                request:request.clone(),accepted:rejection.is_none(),rejection,at:now_ms()};
            tx.put(&target,&RecordedInputDecision {principal:principal.id.clone(),decision:decision.clone()})?;
            append_event(tx,&mut session,Some(&request.turn_id),"input_decided",
                json!({"request_id":request.request_id,"input_id":request.input_id,"accepted":decision.accepted,"rejection":decision.rejection}))?;
            Ok(decision)
        })
    }

    pub fn pending_inputs(&self, principal: &Principal, id: &str) -> Result<Vec<PendingInput>> {
        let session = self.get(principal, id)?;
        Ok(self
            .store
            .scan::<PendingInput>(&key("pending", &[id]))?
            .into_iter()
            .filter(|p| {
                p.response.is_none()
                    && p.generation == session.generation
                    && session.turns.iter().any(|t| {
                        t.id == p.turn_id
                            && !t.status.terminal()
                            && t.status != TurnStatus::Unknown
                            && !t.cancel_requested
                    })
            })
            .collect())
    }

    pub fn revoke_unadmitted(
        &self,
        caller: &Principal,
        automation_id: &str,
        request: &RevokeAutomation,
    ) -> Result<AutomationRevocation> {
        let definition = self
            .config
            .automations
            .get(automation_id)
            .ok_or_else(|| Fault::missing("automation not found"))?;
        if !caller.has_role("scheduler")
            || (!caller.has_role("admin")
                && !caller.sources.contains(&request.invocation.event.source))
            || !definition
                .sources
                .contains(&request.invocation.event.source)
        {
            return Err(Fault::new(
                "forbidden",
                "revocation requires a scheduler granted this automation source",
            ));
        }
        validate_key(&request.request_id)?;
        validate_key(&request.invocation.occurrence_id)?;
        validate_key(&request.invocation.event.id)?;
        if request.invocation.event.schema_version != SCHEMA_VERSION
            || request.reason.trim().is_empty()
            || request.reason.len() > 4096
        {
            return Err(Fault::invalid(
                "revocation requires the original versioned occurrence and a bounded reason",
            ));
        }
        self.store.transaction(|tx| {
            let run_key = key(
                "automation",
                &[automation_id, &request.invocation.occurrence_id],
            );
            let admission = if let Some(run) = tx.get::<AutomationRun>(&run_key)? {
                if run.request != request.invocation {
                    return Err(Fault::conflict(
                        "occurrence already admitted with different input",
                    ));
                }
                Some(run.admission)
            } else {
                let revoked_key = key(
                    "automation_revoked",
                    &[automation_id, &request.invocation.occurrence_id],
                );
                if let Some(old) = tx.get::<RevokedAutomation>(&revoked_key)? {
                    if old.request != *request || old.principal != caller.id {
                        return Err(Fault::conflict(
                            "occurrence already revoked by a different command",
                        ));
                    }
                } else {
                    tx.put(
                        &revoked_key,
                        &RevokedAutomation {
                            principal: caller.id.clone(),
                            request: request.clone(),
                            at: now_ms(),
                        },
                    )?;
                }
                None
            };
            Ok(AutomationRevocation {
                schema_version: SCHEMA_VERSION,
                request_id: request.request_id.clone(),
                automation_id: automation_id.into(),
                occurrence_id: request.invocation.occurrence_id.clone(),
                revoked: admission.is_none(),
                admission,
            })
        })
    }

    pub fn invoke(
        &self,
        caller: &Principal,
        automation_id: &str,
        request: &InvokeAutomation,
    ) -> Result<Admission> {
        let definition = self
            .config
            .automations
            .get(automation_id)
            .ok_or_else(|| Fault::missing("automation not found"))?;
        if !caller.has_role("event") && !caller.has_role("scheduler") {
            return Err(Fault::new(
                "forbidden",
                "automation invocation requires an ingress or scheduler identity",
            ));
        }
        if !caller.has_role("admin") && !caller.sources.contains(&request.event.source) {
            return Err(Fault::new(
                "forbidden",
                "event source not granted to caller",
            ));
        }
        if !definition.enabled || !definition.sources.contains(&request.event.source) {
            return Err(Fault::new(
                "forbidden",
                "automation disabled or source not granted",
            ));
        }
        if request.event.schema_version != SCHEMA_VERSION {
            return Err(Fault::invalid("unsupported event schema version"));
        }
        validate_key(&request.occurrence_id)?;
        validate_key(&request.event.id)?;
        let owner = self
            .config
            .principals
            .iter()
            .find(|p| p.id == definition.principal)
            .ok_or_else(|| Fault::invalid("automation owner is not configured"))?;
        self.store.transaction(|tx| {
            let run_key = key("automation", &[automation_id, &request.occurrence_id]);
            if let Some(old) = tx.get::<AutomationRun>(&run_key)? {
                if old.request != *request {
                    return Err(Fault::conflict(
                        "automation occurrence reused with different event data",
                    ));
                }
                return Ok(old.admission);
            }
            if tx
                .get::<RevokedAutomation>(&key(
                    "automation_revoked",
                    &[automation_id, &request.occurrence_id],
                ))?
                .is_some()
            {
                return Err(Fault::conflict(
                    "automation occurrence was durably revoked before admission",
                ));
            }
            let now = now_ms();
            if request.event.occurred_at > now.saturating_add(120_000)
                || now.saturating_sub(request.event.occurred_at) > definition.max_age_ms
            {
                return Err(Fault::new(
                    "expired",
                    "event outside the automation admission window",
                ));
            }
            let mut active = 0;
            for run in tx.scan::<AutomationRun>(&key("automation", &[automation_id]))? {
                if let Some(turn_id) = run.admission.turn_id {
                    let session = load(tx, &run.admission.session_id)?;
                    if session
                        .turns
                        .iter()
                        .any(|t| t.id == turn_id && !t.status.terminal())
                    {
                        active += 1;
                    }
                }
            }
            if active >= definition.max_active {
                return Err(Fault::new("busy", "automation concurrency limit reached"));
            }
            let request_id = format!(
                "auto-{:x}",
                Sha256::digest(
                    key("invocation", &[automation_id, &request.occurrence_id]).as_bytes()
                )
            );
            let session = match &definition.target {
                SessionTarget::Existing { session_id } => {
                    let session = load(tx, session_id)?;
                    authorize(&session, owner)?;
                    session
                }
                target => {
                    let conversation_key = match target {
                        SessionTarget::PerOccurrence => None,
                        SessionTarget::Conversation { key } => Some(key.clone()),
                        SessionTarget::Incident => Some(format!(
                            "{}:{}",
                            request.event.source,
                            request.event.correlation_id.as_deref().ok_or_else(|| {
                                Fault::invalid("incident routing requires correlation_id")
                            })?
                        )),
                        SessionTarget::Existing { .. } => unreachable!(),
                    };
                    self.create_in(
                        tx,
                        owner,
                        &NewSession {
                            request_id: request_id.clone(),
                            akh: definition.akh.clone(),
                            workspace_profile: definition.workspace_profile.clone(),
                            conversation_key,
                        },
                    )?
                }
            };
            if session.akh != definition.akh
                || session.workspace_profile != definition.workspace_profile
            {
                return Err(Fault::conflict(
                    "automation target has a different Akh or workspace profile",
                ));
            }
            let turn_id = if definition.notify_only {
                let mut session = session.clone();
                append_event(
                    tx,
                    &mut session,
                    None,
                    "external_event",
                    encode(&request.event)?,
                )?;
                None
            } else {
                Some(
                    Self::submit_in(
                        tx,
                        owner,
                        &session.id,
                        &request_id,
                        TurnInput {
                            origin: InputOrigin::Automation {
                                automation_id: automation_id.into(),
                                occurrence_id: request.occurrence_id.clone(),
                            },
                            text: definition.instructions.clone(),
                            data: Some(encode(&request.event)?),
                            artifacts: vec![],
                        },
                    )?
                    .id,
                )
            };
            let admission = Admission {
                session_id: session.id,
                turn_id,
            };
            tx.put(
                &run_key,
                &AutomationRun {
                    request: request.clone(),
                    admission: admission.clone(),
                },
            )?;
            Ok(admission)
        })
    }
}
