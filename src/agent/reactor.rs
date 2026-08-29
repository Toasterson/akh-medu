//! Durable daemon reactor for time- and event-driven agent actions.
//!
//! Unlike Tokio intervals, reactor schedules survive process restarts. A due
//! action is leased before execution, retried with bounded exponential
//! backoff, and completed atomically in the engine metadata store. Event
//! sources (for example a Djehuty thread cursor) use the same execution path as
//! cron and interval schedules.

use std::collections::VecDeque;
use std::str::FromStr;

use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};

use super::error::{AgentError, AgentResult};
use super::trigger::TriggerAction;
use crate::engine::Engine;

const SCHEDULE_PREFIX: &[u8] = b"reactor:schedule:";
const LLM_RESULT_PREFIX: &str = "reactor:llm-result:";
const MAX_PENDING_EVENTS: usize = 256;

fn reactor_error(message: impl Into<String>) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "reactor".into(),
        message: message.into(),
    }
}

/// How a durable schedule becomes due.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScheduleRule {
    /// Run exactly once at the Unix timestamp.
    Once { at: u64 },
    /// Run on a fixed wall-clock cadence anchored at `anchor`.
    Interval { every_seconds: u64, anchor: u64 },
    /// Run according to a UTC cron expression. Five-field POSIX expressions
    /// are accepted and normalized to a zero-seconds expression; the cron
    /// crate's six- and seven-field forms are accepted directly.
    Cron { expression: String },
    /// Run only when a producer publishes an event for `source`.
    Event { source: String },
}

/// What a schedule executes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReactorAction {
    /// Run one real tool-capable model turn through the configured akh backend.
    LlmTurn {
        prompt: String,
        #[serde(default)]
        agent_id: Option<String>,
        /// Offer the agent tool catalog. Disabled by default so an unattended
        /// prompt cannot inherit the standalone akh permission bypass merely
        /// by being scheduled.
        #[serde(default)]
        allow_tools: bool,
        /// Require the autonomous-dispatch postcondition that at least one
        /// mutating tool call succeeded before model completion is accepted.
        #[serde(default)]
        require_mutation: bool,
    },
    /// Respond to a Djehuty thread event with a real model turn, then post the
    /// final answer back to the originating thread with a stable idempotency
    /// key. Board ingestion supplies the triggering message as event payload.
    DjehutyReply {
        thread_id: String,
        #[serde(default)]
        prompt_prefix: String,
        #[serde(default)]
        agent_id: Option<String>,
        #[serde(default)]
        allowed_message_types: Vec<String>,
        #[serde(default = "default_board_prompt_chars")]
        max_prompt_chars: usize,
        #[serde(default = "default_board_reply_chars")]
        max_reply_chars: usize,
        #[serde(default = "default_board_reply_interval")]
        min_reply_interval_seconds: u64,
    },
    /// Reuse an existing deterministic trigger action.
    Trigger { action: TriggerAction },
    /// Inspect a BabyLLM/LoRA training heartbeat and emit only meaningful
    /// state transitions. The deterministic monitor remains the safety path;
    /// an LLM can be scheduled separately to explain the resulting event.
    TrainingMonitor {
        run_id: String,
        status_path: String,
        #[serde(default)]
        control_path: Option<String>,
        #[serde(default = "default_stall_after")]
        stall_after_seconds: u64,
        #[serde(default = "default_progress_steps")]
        progress_report_every_steps: u64,
        #[serde(default = "default_min_disk_free")]
        min_disk_free_bytes: u64,
        #[serde(default)]
        checkpoint_and_stop_on_stall: bool,
    },
}

fn default_stall_after() -> u64 {
    300
}

fn default_progress_steps() -> u64 {
    100
}

fn default_min_disk_free() -> u64 {
    5 * 1024 * 1024 * 1024
}

fn default_board_prompt_chars() -> usize {
    12_000
}

fn default_board_reply_chars() -> usize {
    8_000
}

fn default_board_reply_interval() -> u64 {
    15
}

/// What to do when the daemon returns after a schedule is substantially late.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MisfirePolicy {
    /// Record the missed occurrence and advance to the next one.
    Skip,
    /// Execute one occurrence, never replay an unbounded backlog.
    RunOnce,
}

/// Bounded retry configuration for a reactor action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Retries after the initial attempt.
    pub max_retries: u32,
    pub base_delay_seconds: u64,
    pub max_delay_seconds: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_seconds: 5,
            max_delay_seconds: 300,
        }
    }
}

impl RetryPolicy {
    fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let shift = attempt.saturating_sub(1).min(31);
        self.base_delay_seconds
            .saturating_mul(1_u64 << shift)
            .min(self.max_delay_seconds)
    }
}

/// One durable event waiting to wake an event schedule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReactorEvent {
    pub source: String,
    pub id: String,
    pub observed_at: u64,
    #[serde(default)]
    pub payload: String,
}

/// Mutable, crash-recovered state stored with a schedule.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReactorState {
    pub next_due_at: Option<u64>,
    pub retry_at: Option<u64>,
    pub active_scheduled_for: Option<u64>,
    pub lease_until: Option<u64>,
    pub attempt: u32,
    pub last_started_at: Option<u64>,
    pub last_finished_at: Option<u64>,
    pub last_success_at: Option<u64>,
    pub completed: bool,
    #[serde(default)]
    pub pending_events: VecDeque<ReactorEvent>,
    #[serde(default)]
    pub processed_event_keys: VecDeque<String>,
}

/// A complete durable reactor schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactorSchedule {
    pub id: String,
    pub name: String,
    pub rule: ScheduleRule,
    pub action: ReactorAction,
    pub enabled: bool,
    pub misfire_policy: MisfirePolicy,
    pub misfire_grace_seconds: u64,
    pub lease_seconds: u64,
    pub retry: RetryPolicy,
    #[serde(default)]
    pub state: ReactorState,
}

impl ReactorSchedule {
    /// Construct a schedule and calculate its first due time.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        rule: ScheduleRule,
        action: ReactorAction,
        now: u64,
    ) -> AgentResult<Self> {
        validate_rule(&rule)?;
        validate_action(&rule, &action)?;
        let next_due_at = first_due(&rule, now)?;
        Ok(Self {
            id: id.into(),
            name: name.into(),
            rule,
            action,
            enabled: true,
            misfire_policy: MisfirePolicy::RunOnce,
            misfire_grace_seconds: 60,
            lease_seconds: 900,
            retry: RetryPolicy::default(),
            state: ReactorState {
                next_due_at,
                ..ReactorState::default()
            },
        })
    }

    /// Validate the full durable schedule invariant after deserialization.
    /// Callers that cross a persistence boundary must not treat a structurally
    /// valid JSON object as executable until these semantic checks pass.
    pub(crate) fn validate(&self) -> AgentResult<()> {
        if self.id.trim().is_empty() {
            return Err(reactor_error("schedule id may not be empty"));
        }
        validate_rule(&self.rule)?;
        validate_action(&self.rule, &self.action)
    }
}

/// A leased occurrence returned to the daemon for execution.
#[derive(Debug, Clone)]
pub struct ReactorClaim {
    pub schedule: ReactorSchedule,
    pub run_id: String,
    pub scheduled_for: u64,
    pub attempt: u32,
    pub claimed_at: u64,
    pub event: Option<ReactorEvent>,
}

/// A late occurrence intentionally skipped by policy.
#[derive(Debug, Clone)]
pub struct ReactorMisfire {
    pub schedule_id: String,
    pub schedule_name: String,
    pub scheduled_for: u64,
    pub observed_at: u64,
}

/// Results of polling the durable store once.
#[derive(Debug, Default)]
pub struct ReactorPoll {
    pub claims: Vec<ReactorClaim>,
    pub misfires: Vec<ReactorMisfire>,
}

enum PollSchedule {
    Idle,
    Misfire(ReactorMisfire),
    Claim(ReactorClaim),
}

fn is_poll_candidate(schedule: &ReactorSchedule, now: u64) -> bool {
    if !schedule.enabled
        || schedule.state.completed
        || schedule.state.lease_until.is_some_and(|until| until > now)
    {
        return false;
    }
    if let ReactorAction::DjehutyReply {
        min_reply_interval_seconds,
        ..
    } = &schedule.action
        && schedule
            .state
            .last_success_at
            .is_some_and(|last| last.saturating_add(*min_reply_interval_seconds) > now)
    {
        return false;
    }
    if schedule.state.active_scheduled_for.is_some() {
        return schedule
            .state
            .retry_at
            .is_none_or(|retry_at| retry_at <= now);
    }
    match &schedule.rule {
        ScheduleRule::Event { .. } => !schedule.state.pending_events.is_empty(),
        _ => schedule.state.next_due_at.is_some_and(|due| due <= now),
    }
}

fn poll_schedule(schedule: &mut ReactorSchedule, now: u64) -> AgentResult<PollSchedule> {
    if !schedule.enabled || schedule.state.completed {
        return Ok(PollSchedule::Idle);
    }
    if schedule.state.lease_until.is_some_and(|until| until > now) {
        return Ok(PollSchedule::Idle);
    }
    if let ReactorAction::DjehutyReply {
        min_reply_interval_seconds,
        ..
    } = &schedule.action
        && schedule
            .state
            .last_success_at
            .is_some_and(|last| last.saturating_add(*min_reply_interval_seconds) > now)
    {
        return Ok(PollSchedule::Idle);
    }

    let event = match &schedule.rule {
        ScheduleRule::Event { .. } => schedule.state.pending_events.front().cloned(),
        _ => None,
    };
    let scheduled_for = if let Some(active) = schedule.state.active_scheduled_for {
        if schedule
            .state
            .retry_at
            .is_some_and(|retry_at| retry_at > now)
        {
            return Ok(PollSchedule::Idle);
        }
        active
    } else if let Some(event) = &event {
        event.observed_at
    } else if let Some(due) = schedule.state.next_due_at {
        if due > now {
            return Ok(PollSchedule::Idle);
        }
        due
    } else {
        return Ok(PollSchedule::Idle);
    };

    let lateness = now.saturating_sub(scheduled_for);
    if schedule.misfire_policy == MisfirePolicy::Skip
        && lateness > schedule.misfire_grace_seconds
        && schedule.state.active_scheduled_for.is_none()
    {
        let misfire = ReactorMisfire {
            schedule_id: schedule.id.clone(),
            schedule_name: schedule.name.clone(),
            scheduled_for,
            observed_at: now,
        };
        advance_after_terminal(schedule, scheduled_for, now, event.as_ref())?;
        return Ok(PollSchedule::Misfire(misfire));
    }

    schedule.state.active_scheduled_for = Some(scheduled_for);
    schedule.state.retry_at = None;
    schedule.state.attempt = schedule.state.attempt.saturating_add(1);
    schedule.state.last_started_at = Some(now);
    schedule.state.lease_until = Some(now.saturating_add(schedule.lease_seconds.max(1)));
    let attempt = schedule.state.attempt;
    let run_id = stable_run_id(&schedule.id, scheduled_for, event.as_ref());
    Ok(PollSchedule::Claim(ReactorClaim {
        schedule: schedule.clone(),
        run_id,
        scheduled_for,
        attempt,
        claimed_at: now,
        event,
    }))
}

/// Durable schedule storage and leasing.
pub struct ReactorStore<'a> {
    engine: &'a Engine,
}

impl<'a> ReactorStore<'a> {
    pub fn new(engine: &'a Engine) -> Self {
        Self { engine }
    }

    pub fn list(&self) -> Vec<ReactorSchedule> {
        self.list_checked().unwrap_or_default()
    }

    /// List schedules without hiding durable-store or decoding failures.
    /// Execution paths use this fail-closed form so corruption cannot silently
    /// turn an enabled schedule into a no-op.
    pub fn list_checked(&self) -> AgentResult<Vec<ReactorSchedule>> {
        let entries = self
            .engine
            .store()
            .scan_prefix(SCHEDULE_PREFIX)
            .map_err(|error| AgentError::Engine(Box::new(error.into())))?;
        let mut schedules = Vec::new();
        for (key, value) in entries {
            if value.is_empty() {
                continue;
            }
            let schedule: ReactorSchedule = serde_json::from_slice(&value).map_err(|error| {
                reactor_error(format!(
                    "decode durable schedule {:?}: {error}",
                    String::from_utf8_lossy(&key)
                ))
            })?;
            schedule.validate()?;
            schedules.push(schedule);
        }
        schedules.sort_by(|a: &ReactorSchedule, b| a.id.cmp(&b.id));
        Ok(schedules)
    }

    pub fn get(&self, id: &str) -> Option<ReactorSchedule> {
        let key = schedule_key(id);
        let data = self.engine.store().get_meta(&key).ok()??;
        if data.is_empty() {
            return None;
        }
        serde_json::from_slice(&data).ok()
    }

    fn mutate_schedule<R>(
        &self,
        id: &str,
        mut mutation: impl FnMut(&mut ReactorSchedule) -> AgentResult<R>,
    ) -> AgentResult<R> {
        self.ensure_durable()?;
        let key = schedule_key(id);
        for _ in 0..32 {
            let current = self
                .engine
                .store()
                .get_meta(&key)
                .map_err(|error| AgentError::Engine(Box::new(error.into())))?
                .filter(|bytes| !bytes.is_empty())
                .ok_or_else(|| reactor_error(format!("unknown schedule {id:?}")))?;
            let mut schedule: ReactorSchedule =
                serde_json::from_slice(&current).map_err(|error| {
                    reactor_error(format!("decode durable schedule {id:?}: {error}"))
                })?;
            let result = mutation(&mut schedule)?;
            schedule.validate()?;
            let replacement = serde_json::to_vec(&schedule)
                .map_err(|error| reactor_error(format!("serialize schedule: {error}")))?;
            let swapped = self
                .engine
                .store()
                .compare_and_swap_meta(&key, Some(&current), &replacement)
                .map_err(|error| AgentError::Engine(Box::new(error.into())))?;
            if swapped {
                return Ok(result);
            }
        }
        Err(reactor_error(format!(
            "schedule {id:?} remained contended after 32 atomic retries"
        )))
    }

    pub fn put(&self, schedule: &ReactorSchedule) -> AgentResult<()> {
        self.ensure_durable()?;
        schedule.validate()?;
        let data = serde_json::to_vec(schedule)
            .map_err(|error| reactor_error(format!("serialize schedule: {error}")))?;
        self.engine
            .store()
            .put_meta(&schedule_key(&schedule.id), &data)
            .map_err(|error| AgentError::Engine(Box::new(error.into())))
    }

    pub fn remove(&self, id: &str) -> AgentResult<()> {
        self.ensure_durable()?;
        self.engine
            .store()
            .put_meta(&schedule_key(id), &[])
            .map_err(|error| AgentError::Engine(Box::new(error.into())))
    }

    pub fn set_enabled(&self, id: &str, enabled: bool) -> AgentResult<()> {
        self.mutate_schedule(id, |schedule| {
            schedule.enabled = enabled;
            if !enabled {
                schedule.state.lease_until = None;
            }
            Ok(())
        })
    }

    /// Queue an idempotent event. Duplicate source/id pairs are ignored.
    pub fn wake(&self, id: &str, event: ReactorEvent) -> AgentResult<bool> {
        self.mutate_schedule(id, |schedule| {
            let expected_source = match &schedule.rule {
                ScheduleRule::Event { source } => source,
                _ => return Err(reactor_error("only event schedules can be woken")),
            };
            if expected_source != &event.source {
                return Err(reactor_error(format!(
                    "event source {:?} does not match schedule source {:?}",
                    event.source, expected_source
                )));
            }
            if schedule
                .state
                .pending_events
                .iter()
                .any(|queued| queued.source == event.source && queued.id == event.id)
                || schedule
                    .state
                    .processed_event_keys
                    .iter()
                    .any(|key| key == &event_key(&event))
            {
                return Ok(false);
            }
            if schedule.state.pending_events.len() >= MAX_PENDING_EVENTS {
                return Err(reactor_error("event queue is full"));
            }
            schedule.state.pending_events.push_back(event.clone());
            Ok(true)
        })
    }

    /// Lease every currently-due occurrence. A lease is persisted before the
    /// claim is returned, so another daemon (or a restarted one) cannot run it
    /// until the lease expires.
    pub fn poll_due(&self, now: u64) -> AgentResult<ReactorPoll> {
        let mut poll = ReactorPoll::default();
        let schedule_ids: Vec<String> = self
            .list_checked()?
            .into_iter()
            .filter(|schedule| is_poll_candidate(schedule, now))
            .map(|schedule| schedule.id)
            .collect();
        for id in schedule_ids {
            let outcome = self.mutate_schedule(&id, |schedule| poll_schedule(schedule, now))?;
            match outcome {
                PollSchedule::Idle => {}
                PollSchedule::Misfire(misfire) => poll.misfires.push(misfire),
                PollSchedule::Claim(claim) => poll.claims.push(claim),
            }
        }
        Ok(poll)
    }

    /// Finish one leased attempt and either advance the schedule or persist a
    /// bounded retry due time.
    pub fn complete(&self, claim: &ReactorClaim, now: u64, success: bool) -> AgentResult<()> {
        self.mutate_schedule(&claim.schedule.id, |schedule| {
            if schedule.state.active_scheduled_for != Some(claim.scheduled_for)
                || schedule.state.attempt != claim.attempt
            {
                return Err(reactor_error("stale reactor claim completion rejected"));
            }
            schedule.state.last_finished_at = Some(now);
            schedule.state.lease_until = None;
            if success {
                schedule.state.last_success_at = Some(now);
                advance_after_terminal(schedule, claim.scheduled_for, now, claim.event.as_ref())?;
            } else if claim.attempt <= schedule.retry.max_retries {
                let delay = schedule.retry.delay_for_attempt(claim.attempt);
                schedule.state.retry_at = Some(now.saturating_add(delay));
            } else {
                advance_after_terminal(schedule, claim.scheduled_for, now, claim.event.as_ref())?;
            }
            Ok(())
        })
    }

    /// Release a claim that could not begin because the live agent lock was
    /// contended. This is not an action failure: it preserves the occurrence,
    /// does not consume retry budget, and makes it eligible after `delay`.
    pub fn defer_claim(&self, claim: &ReactorClaim, now: u64, delay: u64) -> AgentResult<()> {
        self.mutate_schedule(&claim.schedule.id, |schedule| {
            if schedule.state.active_scheduled_for != Some(claim.scheduled_for)
                || schedule.state.attempt != claim.attempt
            {
                return Err(reactor_error("stale reactor claim deferral rejected"));
            }
            schedule.state.lease_until = None;
            schedule.state.retry_at = Some(now.saturating_add(delay.max(1)));
            schedule.state.attempt = schedule.state.attempt.saturating_sub(1);
            Ok(())
        })
    }

    /// Cache the bounded output of a tool-free LLM occurrence before schedule
    /// completion. A completion-write failure can then replay the durable
    /// result without paying for or varying a second model turn.
    pub fn cache_llm_result(&self, run_id: &str, summary: &str) -> AgentResult<()> {
        self.ensure_durable()?;
        if run_id.is_empty() || summary.is_empty() {
            return Err(reactor_error(
                "cached LLM run id and summary may not be empty",
            ));
        }
        self.engine
            .store()
            .put_meta(
                format!("{LLM_RESULT_PREFIX}{run_id}").as_bytes(),
                summary.as_bytes(),
            )
            .map_err(|error| AgentError::Engine(Box::new(error.into())))
    }

    pub fn cached_llm_result(&self, run_id: &str) -> AgentResult<Option<String>> {
        self.ensure_durable()?;
        self.engine
            .store()
            .get_meta(format!("{LLM_RESULT_PREFIX}{run_id}").as_bytes())
            .map_err(|error| AgentError::Engine(Box::new(error.into())))?
            .map(|bytes| {
                String::from_utf8(bytes)
                    .map_err(|error| reactor_error(format!("decode cached LLM result: {error}")))
            })
            .transpose()
    }

    fn ensure_durable(&self) -> AgentResult<()> {
        if self.engine.store().durable.is_none() {
            return Err(reactor_error(
                "durable reactor requires an engine data directory",
            ));
        }
        Ok(())
    }
}

fn schedule_key(id: &str) -> Vec<u8> {
    format!("reactor:schedule:{id}").into_bytes()
}

fn stable_run_id(id: &str, scheduled_for: u64, event: Option<&ReactorEvent>) -> String {
    let event_id = event.map(|item| item.id.as_str()).unwrap_or("");
    let digest = blake3::hash(format!("{id}\0{scheduled_for}\0{event_id}").as_bytes());
    digest.to_hex()[..24].to_string()
}

fn event_key(event: &ReactorEvent) -> String {
    format!("{}\0{}", event.source, event.id)
}

fn validate_rule(rule: &ScheduleRule) -> AgentResult<()> {
    match rule {
        ScheduleRule::Once { .. } => Ok(()),
        ScheduleRule::Interval { every_seconds, .. } if *every_seconds == 0 => {
            Err(reactor_error("interval must be at least one second"))
        }
        ScheduleRule::Interval { .. } => Ok(()),
        ScheduleRule::Cron { expression } => parse_cron(expression).map(|_| ()),
        ScheduleRule::Event { source } if source.trim().is_empty() => {
            Err(reactor_error("event source may not be empty"))
        }
        ScheduleRule::Event { .. } => Ok(()),
    }
}

fn validate_action(rule: &ScheduleRule, action: &ReactorAction) -> AgentResult<()> {
    if let ReactorAction::LlmTurn {
        allow_tools,
        require_mutation,
        ..
    } = action
        && *require_mutation
        && !*allow_tools
    {
        return Err(reactor_error(
            "require_mutation requires allow_tools for scheduled LLM turns",
        ));
    }
    let ReactorAction::DjehutyReply {
        thread_id,
        max_prompt_chars,
        max_reply_chars,
        ..
    } = action
    else {
        return Ok(());
    };
    let expected = format!("djehuty:thread:{thread_id}");
    if !matches!(rule, ScheduleRule::Event { source } if source == &expected) {
        return Err(reactor_error(format!(
            "djehuty_reply requires event source {expected:?}"
        )));
    }
    uuid::Uuid::parse_str(thread_id)
        .map_err(|error| reactor_error(format!("invalid Djehuty thread id: {error}")))?;
    if *max_prompt_chars == 0 || *max_reply_chars == 0 {
        return Err(reactor_error(
            "Djehuty prompt and reply limits must be greater than zero",
        ));
    }
    Ok(())
}

fn normalize_cron(expression: &str) -> String {
    if expression.split_whitespace().count() == 5 {
        format!("0 {expression}")
    } else {
        expression.to_string()
    }
}

fn parse_cron(expression: &str) -> AgentResult<cron::Schedule> {
    cron::Schedule::from_str(&normalize_cron(expression))
        .map_err(|error| reactor_error(format!("invalid cron expression: {error}")))
}

fn first_due(rule: &ScheduleRule, now: u64) -> AgentResult<Option<u64>> {
    match rule {
        ScheduleRule::Once { at } => Ok(Some(*at)),
        ScheduleRule::Interval {
            every_seconds,
            anchor,
        } => {
            if now <= *anchor {
                Ok(Some(*anchor))
            } else {
                let elapsed = now - anchor;
                let steps = elapsed / every_seconds + u64::from(elapsed % every_seconds != 0);
                Ok(Some(
                    anchor.saturating_add(steps.saturating_mul(*every_seconds)),
                ))
            }
        }
        ScheduleRule::Cron { expression } => next_cron(expression, now.saturating_sub(1)),
        ScheduleRule::Event { .. } => Ok(None),
    }
}

fn next_due_after(rule: &ScheduleRule, after: u64) -> AgentResult<Option<u64>> {
    match rule {
        ScheduleRule::Once { .. } | ScheduleRule::Event { .. } => Ok(None),
        ScheduleRule::Interval {
            every_seconds,
            anchor,
        } => {
            let elapsed = after.saturating_sub(*anchor);
            let steps = elapsed / every_seconds + 1;
            Ok(Some(
                anchor.saturating_add(steps.saturating_mul(*every_seconds)),
            ))
        }
        ScheduleRule::Cron { expression } => next_cron(expression, after),
    }
}

fn next_cron(expression: &str, after: u64) -> AgentResult<Option<u64>> {
    let schedule = parse_cron(expression)?;
    let after_i64 = i64::try_from(after).map_err(|_| reactor_error("cron timestamp overflow"))?;
    let instant = Utc
        .timestamp_opt(after_i64, 0)
        .single()
        .ok_or_else(|| reactor_error("invalid cron timestamp"))?;
    Ok(schedule
        .after(&instant)
        .next()
        .and_then(|next| u64::try_from(next.timestamp()).ok()))
}

fn advance_after_terminal(
    schedule: &mut ReactorSchedule,
    scheduled_for: u64,
    now: u64,
    event: Option<&ReactorEvent>,
) -> AgentResult<()> {
    if let Some(event) = event
        && schedule
            .state
            .pending_events
            .front()
            .is_some_and(|queued| queued.source == event.source && queued.id == event.id)
    {
        schedule.state.pending_events.pop_front();
        schedule
            .state
            .processed_event_keys
            .push_back(event_key(event));
        while schedule.state.processed_event_keys.len() > MAX_PENDING_EVENTS {
            schedule.state.processed_event_keys.pop_front();
        }
    }
    schedule.state.active_scheduled_for = None;
    schedule.state.retry_at = None;
    schedule.state.lease_until = None;
    schedule.state.attempt = 0;
    schedule.state.next_due_at = next_due_after(&schedule.rule, scheduled_for.max(now))?;
    if matches!(schedule.rule, ScheduleRule::Once { .. }) {
        schedule.state.completed = true;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    fn engine() -> Engine {
        let data_dir = tempfile::tempdir().unwrap().keep();
        Engine::new(EngineConfig {
            data_dir: Some(data_dir),
            ..EngineConfig::default()
        })
        .unwrap()
    }

    fn action() -> ReactorAction {
        ReactorAction::LlmTurn {
            prompt: "summarize status".into(),
            agent_id: None,
            allow_tools: false,
            require_mutation: false,
        }
    }

    #[test]
    fn five_field_cron_is_accepted_and_calculated_in_utc() {
        let schedule = ReactorSchedule::new(
            "daily",
            "daily",
            ScheduleRule::Cron {
                expression: "0 6 * * *".into(),
            },
            action(),
            1_700_000_000,
        )
        .unwrap();
        assert!(schedule.state.next_due_at.unwrap() > 1_700_000_000);
    }

    #[test]
    fn lease_is_durable_and_retry_is_bounded() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let mut schedule = ReactorSchedule::new(
            "s1",
            "interval",
            ScheduleRule::Interval {
                every_seconds: 60,
                anchor: 100,
            },
            action(),
            100,
        )
        .unwrap();
        schedule.retry = RetryPolicy {
            max_retries: 1,
            base_delay_seconds: 5,
            max_delay_seconds: 5,
        };
        store.put(&schedule).unwrap();

        let first = store.poll_due(100).unwrap().claims.remove(0);
        assert!(store.poll_due(101).unwrap().claims.is_empty());
        store.complete(&first, 102, false).unwrap();
        assert!(store.poll_due(106).unwrap().claims.is_empty());
        let retry = store.poll_due(107).unwrap().claims.remove(0);
        assert_eq!(retry.run_id, first.run_id);
        assert_eq!(retry.attempt, 2);
        store.complete(&retry, 108, false).unwrap();
        let stored = store.get("s1").unwrap();
        assert_eq!(stored.state.next_due_at, Some(160));
        assert_eq!(stored.state.attempt, 0);
    }

    #[test]
    fn event_wakes_are_idempotent_and_ordered() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let schedule = ReactorSchedule::new(
            "board",
            "board",
            ScheduleRule::Event {
                source: "djehuty:thread:7".into(),
            },
            action(),
            10,
        )
        .unwrap();
        store.put(&schedule).unwrap();
        let event = ReactorEvent {
            source: "djehuty:thread:7".into(),
            id: "m1".into(),
            observed_at: 11,
            payload: "hello".into(),
        };
        assert!(store.wake("board", event.clone()).unwrap());
        assert!(!store.wake("board", event).unwrap());
        let claim = store.poll_due(11).unwrap().claims.remove(0);
        assert_eq!(claim.event.as_ref().unwrap().id, "m1");
        store.complete(&claim, 12, true).unwrap();
        assert!(store.poll_due(12).unwrap().claims.is_empty());
        assert!(
            !store
                .wake(
                    "board",
                    ReactorEvent {
                        source: "djehuty:thread:7".into(),
                        id: "m1".into(),
                        observed_at: 13,
                        payload: "duplicate".into(),
                    },
                )
                .unwrap()
        );
    }

    #[test]
    fn concurrent_wakes_preserve_every_event() {
        let engine = std::sync::Arc::new(engine());
        let schedule = ReactorSchedule::new(
            "concurrent-board",
            "concurrent board",
            ScheduleRule::Event {
                source: "djehuty:thread:atomic".into(),
            },
            action(),
            10,
        )
        .unwrap();
        ReactorStore::new(&engine).put(&schedule).unwrap();

        let mut workers = Vec::new();
        for worker in 0..8 {
            let engine = std::sync::Arc::clone(&engine);
            workers.push(std::thread::spawn(move || {
                for offset in 0..8 {
                    ReactorStore::new(&engine)
                        .wake(
                            "concurrent-board",
                            ReactorEvent {
                                source: "djehuty:thread:atomic".into(),
                                id: format!("{worker}-{offset}"),
                                observed_at: 11,
                                payload: "{}".into(),
                            },
                        )
                        .unwrap();
                }
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }

        let stored = ReactorStore::new(&engine).get("concurrent-board").unwrap();
        assert_eq!(stored.state.pending_events.len(), 64);
    }

    #[test]
    fn concurrent_pollers_issue_one_lease() {
        let engine = std::sync::Arc::new(engine());
        let schedule = ReactorSchedule::new(
            "one-lease",
            "one lease",
            ScheduleRule::Once { at: 10 },
            action(),
            0,
        )
        .unwrap();
        ReactorStore::new(&engine).put(&schedule).unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let engine = std::sync::Arc::clone(&engine);
            let barrier = std::sync::Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                ReactorStore::new(&engine)
                    .poll_due(10)
                    .unwrap()
                    .claims
                    .len()
            }));
        }
        barrier.wait();
        let claims: usize = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .sum();
        assert_eq!(claims, 1);
    }

    #[test]
    fn corrupt_durable_schedule_fails_poll_closed() {
        let engine = engine();
        engine
            .store()
            .put_meta(b"reactor:schedule:corrupt", b"not-json")
            .unwrap();
        let error = ReactorStore::new(&engine).poll_due(10).unwrap_err();
        assert!(error.to_string().contains("decode durable schedule"));
    }

    #[test]
    fn lock_deferral_preserves_occurrence_and_retry_budget() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let schedule =
            ReactorSchedule::new("defer", "defer", ScheduleRule::Once { at: 10 }, action(), 0)
                .unwrap();
        store.put(&schedule).unwrap();
        let claim = store.poll_due(10).unwrap().claims.remove(0);
        assert_eq!(claim.attempt, 1);
        store.defer_claim(&claim, 10, 5).unwrap();
        let deferred = store.get("defer").unwrap();
        assert_eq!(deferred.state.attempt, 0);
        assert_eq!(deferred.state.retry_at, Some(15));
        assert!(store.poll_due(14).unwrap().claims.is_empty());
        assert_eq!(store.poll_due(15).unwrap().claims.remove(0).attempt, 1);
    }

    #[test]
    fn skip_policy_records_misfire_without_running() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let mut schedule = ReactorSchedule::new(
            "skip",
            "skip",
            ScheduleRule::Interval {
                every_seconds: 60,
                anchor: 100,
            },
            action(),
            100,
        )
        .unwrap();
        schedule.misfire_policy = MisfirePolicy::Skip;
        schedule.misfire_grace_seconds = 5;
        store.put(&schedule).unwrap();
        let poll = store.poll_due(200).unwrap();
        assert!(poll.claims.is_empty());
        assert_eq!(poll.misfires.len(), 1);
        assert_eq!(store.get("skip").unwrap().state.next_due_at, Some(220));
    }
}
