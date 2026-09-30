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
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::error::{AgentError, AgentResult};
use super::trigger::TriggerAction;
use crate::engine::Engine;

const SCHEDULE_PREFIX: &[u8] = b"reactor:schedule:";
const LLM_RESULT_PREFIX: &str = "reactor:llm-result:";
const MAX_PENDING_EVENTS: usize = 256;

/// ADR 105 rule 5: a schedule whose run outcome repeats this many times in a
/// row without progress pauses itself and reports instead of running again.
pub const NO_PROGRESS_TRIP_WIRE: u32 = 10;

/// ADR 105 governance violations of a reactor schedule (Phase 53d).
#[derive(Debug, Clone, Error, Diagnostic, PartialEq, Eq)]
pub enum ReactorGovernanceError {
    #[error(
        "schedule `{schedule}` offers tools to an unattended LLM turn without an exit condition"
    )]
    #[diagnostic(
        code(akh::reactor::exit_condition_required),
        help(
            "ADR 105 rule 1: an autonomous run states what \"done\" and \"failed\" look like. Set \
             `action.exit_condition` to a machine-checkable statement (for example \"the thread \
             has a coordination/status@v1 message with state complete for today\"), or set \
             `allow_tools = false` for a tool-free turn. Stored schedules without it are paused, \
             not deleted: fix the action with `reactor_manage` operation `update`, then enable it."
        )
    )]
    ExitConditionRequired { schedule: String },

    #[error(
        "schedule `{schedule}` names report_thread `{thread}`, which is not a board thread UUID"
    )]
    #[diagnostic(
        code(akh::reactor::report_thread),
        help(
            "`report_thread` is the Agent Board thread UUID that receives the run's \
             coordination/status@v1 reports (ADR 105 rule 3). Copy it from the board, or omit it \
             to keep reports in the akh's own episodes."
        )
    )]
    InvalidReportThread { schedule: String, thread: String },
}

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
        /// ADR 105 rule 1: what "done" and "failed" look like for this run.
        /// Required when `allow_tools` is set. Schedules stored before this
        /// field existed still load; a tool-capable one without it is paused
        /// by the reactor (see [`PauseReason::MissingExitCondition`]).
        #[serde(default, skip_serializing_if = "String::is_empty")]
        exit_condition: String,
        /// Agent Board thread (UUID) that receives a `coordination/status@v1`
        /// message after every run (ADR 105 rule 3).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        report_thread: Option<String>,
        /// Tools this wake always needs (ADR 123). A wake's catalog is
        /// relevance-scoped to the model profile's `max_tools_per_wake`;
        /// these tools, and any tool the prompt or exit condition names by
        /// its registered name, are offered on top of that.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        required_tools: Vec<String>,
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

/// Why the reactor paused a schedule on its own (ADR 105).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PauseReason {
    /// A tool-capable `LlmTurn` stored before `exit_condition` existed.
    MissingExitCondition,
    /// ADR 105 rule 5: the same no-progress outcome repeated
    /// [`NO_PROGRESS_TRIP_WIRE`] times in a row.
    NoProgressTripWire { repeats: u32, outcome: String },
}

/// A self-imposed pause. Cleared by enabling the schedule again.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchedulePause {
    pub at: u64,
    pub reason: PauseReason,
}

/// The classified outcome of one run, fed to the ADR 105 trip-wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    /// The run changed something (a successful mutating tool call).
    pub progress: bool,
    /// Stable description of the outcome; identical no-progress outcomes
    /// share it.
    pub fingerprint: String,
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
    /// Set when the reactor paused the schedule itself (ADR 105).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause: Option<SchedulePause>,
    /// Consecutive runs that ended with the same no-progress outcome.
    #[serde(default)]
    pub no_progress_streak: u32,
    /// Fingerprint of the last no-progress outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_no_progress: Option<String>,
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
    /// Event schedules only: keep just the newest pending event. For
    /// replaceable state notifications, never commands or delegations.
    #[serde(default)]
    pub coalesce: bool,
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
        let id = id.into();
        validate_rule(&rule)?;
        validate_action(&rule, &action)?;
        if let Some(violation) = governance_violation(&id, &action) {
            return Err(violation.into());
        }
        let next_due_at = first_due(&rule, now)?;
        Ok(Self {
            id,
            name: name.into(),
            rule,
            action,
            enabled: true,
            misfire_policy: MisfirePolicy::RunOnce,
            misfire_grace_seconds: 60,
            lease_seconds: 900,
            retry: RetryPolicy::default(),
            coalesce: false,
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
        if self.id.chars().any(char::is_control) {
            return Err(reactor_error(
                "schedule id may not contain control characters",
            ));
        }
        validate_rule(&self.rule)?;
        validate_action(&self.rule, &self.action)
    }

    /// The strict admission check for new or edited schedules: the durable
    /// invariant plus ADR 105 governance. Stored schedules are loaded with
    /// [`Self::validate`] only, so a legacy schedule still loads and is paused
    /// by the reactor instead of breaking every other schedule.
    pub fn validate_for_admission(&self) -> AgentResult<()> {
        self.validate()?;
        match self.governance_violation() {
            Some(violation) => Err(violation.into()),
            None => Ok(()),
        }
    }

    /// Why the reactor would refuse to run this schedule, if at all.
    pub fn governance_violation(&self) -> Option<ReactorGovernanceError> {
        governance_violation(&self.id, &self.action)
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

impl ReactorClaim {
    /// The prompt an [`ReactorAction::LlmTurn`] claim runs with, `None` for
    /// other actions.
    ///
    /// The configured prompt and exit condition are trusted (the operator or
    /// the akh wrote them). When an event woke the schedule, its payload is
    /// appended as untrusted data with exactly the framing session turns use
    /// ([`akh_session_types::TurnInput::provider_prompt`]), so an event field
    /// can never read as an instruction or an approval.
    pub fn llm_turn_prompt(&self) -> Option<String> {
        let ReactorAction::LlmTurn {
            prompt,
            exit_condition,
            ..
        } = &self.schedule.action
        else {
            return None;
        };
        let text = if exit_condition.trim().is_empty() {
            prompt.clone()
        } else {
            format!(
                "{prompt}\n\nExit condition (ADR 105): {}",
                exit_condition.trim()
            )
        };
        let data = self.event.as_ref().map(|event| {
            serde_json::from_str::<serde_json::Value>(&event.payload)
                .unwrap_or_else(|_| serde_json::Value::String(event.payload.clone()))
        });
        Some(
            akh_session_types::TurnInput {
                origin: akh_session_types::InputOrigin::Automation {
                    automation_id: self.schedule.id.clone(),
                    occurrence_id: self.run_id.clone(),
                },
                text,
                data,
                artifacts: Vec::new(),
            }
            .provider_prompt(),
        )
    }
}

/// A late occurrence intentionally skipped by policy.
#[derive(Debug, Clone)]
pub struct ReactorMisfire {
    pub schedule_id: String,
    pub schedule_name: String,
    pub scheduled_for: u64,
    pub observed_at: u64,
}

/// A schedule the reactor paused while polling (ADR 105).
#[derive(Debug, Clone)]
pub struct ReactorPaused {
    pub schedule_id: String,
    pub schedule_name: String,
    pub pause: SchedulePause,
    pub report_thread: Option<String>,
}

/// Results of polling the durable store once.
#[derive(Debug, Default)]
pub struct ReactorPoll {
    pub claims: Vec<ReactorClaim>,
    pub misfires: Vec<ReactorMisfire>,
    pub paused: Vec<ReactorPaused>,
}

enum PollSchedule {
    Idle,
    Misfire(ReactorMisfire),
    Claim(ReactorClaim),
    Paused(ReactorPaused),
}

fn report_thread_of(action: &ReactorAction) -> Option<String> {
    match action {
        ReactorAction::LlmTurn { report_thread, .. } => report_thread.clone(),
        _ => None,
    }
}

/// Disable `schedule` with a recorded reason and drop its lease.
fn pause_schedule(schedule: &mut ReactorSchedule, reason: PauseReason, now: u64) -> SchedulePause {
    let pause = SchedulePause { at: now, reason };
    schedule.enabled = false;
    schedule.state.lease_until = None;
    schedule.state.pause = Some(pause.clone());
    pause
}

fn is_poll_candidate(schedule: &ReactorSchedule, now: u64) -> bool {
    // An enabled schedule that breaks governance must reach `poll_schedule`
    // once, so it is paused durably and reported rather than run.
    if schedule.enabled && !schedule.state.completed && schedule.governance_violation().is_some() {
        return true;
    }
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
    if let Some(ReactorGovernanceError::ExitConditionRequired { .. }) =
        schedule.governance_violation()
    {
        let pause = pause_schedule(schedule, PauseReason::MissingExitCondition, now);
        return Ok(PollSchedule::Paused(ReactorPaused {
            schedule_id: schedule.id.clone(),
            schedule_name: schedule.name.clone(),
            pause,
            report_thread: report_thread_of(&schedule.action),
        }));
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

    /// Enable or disable a schedule. Enabling clears a self-imposed pause
    /// and the trip-wire streak, and refuses a schedule that still breaks
    /// ADR 105 governance.
    pub fn set_enabled(&self, id: &str, enabled: bool) -> AgentResult<()> {
        self.mutate_schedule(id, |schedule| {
            if enabled && let Some(violation) = schedule.governance_violation() {
                return Err(violation.into());
            }
            schedule.enabled = enabled;
            if enabled {
                schedule.state.pause = None;
                schedule.state.no_progress_streak = 0;
                schedule.state.last_no_progress = None;
            } else {
                schedule.state.lease_until = None;
            }
            Ok(())
        })
    }

    /// Replace a schedule's action (and optionally its coalescing), keeping
    /// its durable state. The result is disabled: re-enabling is the
    /// operator gate, as for newly added schedules.
    pub fn update_action(
        &self,
        id: &str,
        action: ReactorAction,
        coalesce: Option<bool>,
    ) -> AgentResult<()> {
        self.mutate_schedule(id, |schedule| {
            validate_action(&schedule.rule, &action)?;
            if let Some(violation) = governance_violation(&schedule.id, &action) {
                return Err(violation.into());
            }
            schedule.action = action.clone();
            if let Some(coalesce) = coalesce {
                schedule.coalesce = coalesce;
            }
            schedule.enabled = false;
            schedule.state.lease_until = None;
            Ok(())
        })
    }

    /// Feed one run outcome to the ADR 105 trip-wire. Returns the pause when
    /// this outcome tripped it: the schedule is then disabled durably.
    pub fn record_outcome(
        &self,
        id: &str,
        outcome: &RunOutcome,
        now: u64,
    ) -> AgentResult<Option<SchedulePause>> {
        self.mutate_schedule(id, |schedule| {
            let state = &mut schedule.state;
            if outcome.progress {
                state.no_progress_streak = 0;
                state.last_no_progress = None;
                return Ok(None);
            }
            if state.last_no_progress.as_deref() == Some(outcome.fingerprint.as_str()) {
                state.no_progress_streak = state.no_progress_streak.saturating_add(1);
            } else {
                state.no_progress_streak = 1;
                state.last_no_progress = Some(outcome.fingerprint.clone());
            }
            if state.no_progress_streak < NO_PROGRESS_TRIP_WIRE {
                return Ok(None);
            }
            let repeats = state.no_progress_streak;
            Ok(Some(pause_schedule(
                schedule,
                PauseReason::NoProgressTripWire {
                    repeats,
                    outcome: outcome.fingerprint.clone(),
                },
                now,
            )))
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
            if schedule.coalesce {
                coalesce_pending(schedule);
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
                PollSchedule::Paused(paused) => poll.paused.push(paused),
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

/// Drop every pending event a newer one replaces. The front event survives
/// while an occurrence for it is in flight (it is being run or retried, and
/// completion pops it by identity). Superseded events count as processed, so
/// a replay of one is still a duplicate.
fn coalesce_pending(schedule: &mut ReactorSchedule) {
    let keep_front = schedule.state.active_scheduled_for.is_some();
    let state = &mut schedule.state;
    let in_flight = if keep_front {
        state.pending_events.pop_front()
    } else {
        None
    };
    for superseded in state.pending_events.drain(..) {
        state.processed_event_keys.push_back(event_key(&superseded));
    }
    while state.processed_event_keys.len() > MAX_PENDING_EVENTS {
        state.processed_event_keys.pop_front();
    }
    if let Some(front) = in_flight {
        state.pending_events.push_back(front);
    }
}

/// ADR 105: the first governance rule `action` breaks, if any.
fn governance_violation(id: &str, action: &ReactorAction) -> Option<ReactorGovernanceError> {
    let ReactorAction::LlmTurn {
        allow_tools,
        exit_condition,
        report_thread,
        ..
    } = action
    else {
        return None;
    };
    if *allow_tools && exit_condition.trim().is_empty() {
        return Some(ReactorGovernanceError::ExitConditionRequired {
            schedule: id.to_string(),
        });
    }
    if let Some(thread) = report_thread
        && uuid::Uuid::parse_str(thread).is_err()
    {
        return Some(ReactorGovernanceError::InvalidReportThread {
            schedule: id.to_string(),
            thread: thread.clone(),
        });
    }
    None
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
            exit_condition: String::new(),
            report_thread: None,
            required_tools: Vec::new(),
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

    // ── Phase 53d: coalescing and ADR 105 governance ──

    fn event_schedule(id: &str, source: &str) -> ReactorSchedule {
        ReactorSchedule::new(
            id,
            id,
            ScheduleRule::Event {
                source: source.into(),
            },
            action(),
            10,
        )
        .unwrap()
    }

    fn event(source: &str, id: &str, observed_at: u64) -> ReactorEvent {
        ReactorEvent {
            source: source.into(),
            id: id.into(),
            observed_at,
            payload: format!("{{\"id\":\"{id}\"}}"),
        }
    }

    #[test]
    fn coalescing_schedule_keeps_only_the_newest_pending_event() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let mut schedule = event_schedule("state", "observer/state");
        schedule.coalesce = true;
        store.put(&schedule).unwrap();
        for (index, id) in ["a", "b", "c"].into_iter().enumerate() {
            assert!(
                store
                    .wake("state", event("observer/state", id, 11 + index as u64))
                    .unwrap()
            );
        }
        let stored = store.get("state").unwrap();
        let pending: Vec<_> = stored
            .state
            .pending_events
            .iter()
            .map(|e| e.id.as_str())
            .collect();
        assert_eq!(pending, ["c"]);
        // A superseded event replayed later is still a duplicate.
        assert!(
            !store
                .wake("state", event("observer/state", "a", 20))
                .unwrap()
        );
        let claim = store.poll_due(20).unwrap().claims.remove(0);
        assert_eq!(claim.event.unwrap().id, "c");
    }

    #[test]
    fn coalescing_never_drops_the_event_of_an_in_flight_occurrence() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let mut schedule = event_schedule("state", "observer/state");
        schedule.coalesce = true;
        store.put(&schedule).unwrap();
        store
            .wake("state", event("observer/state", "running", 11))
            .unwrap();
        let claim = store.poll_due(11).unwrap().claims.remove(0);
        store
            .wake("state", event("observer/state", "b", 12))
            .unwrap();
        store
            .wake("state", event("observer/state", "c", 13))
            .unwrap();
        let pending: Vec<_> = store
            .get("state")
            .unwrap()
            .state
            .pending_events
            .iter()
            .map(|e| e.id.clone())
            .collect();
        assert_eq!(pending, ["running", "c"]);
        store.complete(&claim, 14, true).unwrap();
        let next = store.poll_due(15).unwrap().claims.remove(0);
        assert_eq!(next.event.unwrap().id, "c");
    }

    fn tool_turn(exit_condition: &str) -> ReactorAction {
        ReactorAction::LlmTurn {
            prompt: "work the queue".into(),
            agent_id: None,
            allow_tools: true,
            require_mutation: false,
            exit_condition: exit_condition.into(),
            report_thread: None,
            required_tools: Vec::new(),
        }
    }

    #[test]
    fn tool_capable_turn_requires_an_exit_condition() {
        let error = ReactorSchedule::new(
            "ungoverned",
            "ungoverned",
            ScheduleRule::Once { at: 10 },
            tool_turn("  "),
            0,
        )
        .unwrap_err();
        let AgentError::ReactorGovernance(inner) = &error else {
            panic!("expected a governance diagnostic, got {error:?}");
        };
        assert_eq!(
            inner.code().unwrap().to_string(),
            "akh::reactor::exit_condition_required"
        );
        assert!(inner.help().is_some());
        ReactorSchedule::new(
            "governed",
            "governed",
            ScheduleRule::Once { at: 10 },
            tool_turn("queue empty or 3 items closed"),
            0,
        )
        .unwrap();
        let bad_thread = ReactorAction::LlmTurn {
            prompt: "p".into(),
            agent_id: None,
            allow_tools: false,
            require_mutation: false,
            exit_condition: String::new(),
            report_thread: Some("not-a-uuid".into()),
            required_tools: Vec::new(),
        };
        assert!(matches!(
            ReactorSchedule::new("t", "t", ScheduleRule::Once { at: 1 }, bad_thread, 0),
            Err(AgentError::ReactorGovernance(
                ReactorGovernanceError::InvalidReportThread { .. }
            ))
        ));
    }

    #[test]
    fn legacy_ungoverned_schedule_loads_and_is_paused_not_deleted() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let mut value = serde_json::to_value(
            ReactorSchedule::new(
                "legacy",
                "legacy",
                ScheduleRule::Once { at: 10 },
                tool_turn("placeholder"),
                0,
            )
            .unwrap(),
        )
        .unwrap();
        // The shape stored before Phase 53d: no exit_condition at all.
        value["action"]
            .as_object_mut()
            .unwrap()
            .remove("exit_condition");
        engine
            .store()
            .put_meta(
                b"reactor:schedule:legacy",
                &serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        assert_eq!(store.list_checked().unwrap().len(), 1, "it still loads");

        let poll = store.poll_due(10).unwrap();
        assert!(poll.claims.is_empty(), "never runs ungoverned");
        assert_eq!(poll.paused.len(), 1);
        assert_eq!(
            poll.paused[0].pause.reason,
            PauseReason::MissingExitCondition
        );
        let stored = store.get("legacy").unwrap();
        assert!(!stored.enabled);
        assert!(stored.state.pause.is_some());
        assert!(store.poll_due(11).unwrap().paused.is_empty(), "paused once");

        // Enabling is refused until the action is governed.
        assert!(matches!(
            store.set_enabled("legacy", true),
            Err(AgentError::ReactorGovernance(_))
        ));
        store
            .update_action("legacy", tool_turn("backlog empty"), None)
            .unwrap();
        store.set_enabled("legacy", true).unwrap();
        let fixed = store.get("legacy").unwrap();
        assert!(fixed.enabled && fixed.state.pause.is_none());
        assert_eq!(store.poll_due(12).unwrap().claims.len(), 1);
    }

    #[test]
    fn trip_wire_pauses_after_ten_identical_no_progress_outcomes() {
        let engine = engine();
        let store = ReactorStore::new(&engine);
        let schedule = ReactorSchedule::new(
            "loop",
            "loop",
            ScheduleRule::Interval {
                every_seconds: 60,
                anchor: 0,
            },
            action(),
            0,
        )
        .unwrap();
        store.put(&schedule).unwrap();
        let stuck = RunOutcome {
            progress: false,
            fingerprint: "failed: provider 429".into(),
        };
        for _ in 0..9 {
            assert!(store.record_outcome("loop", &stuck, 1).unwrap().is_none());
        }
        // Progress or a different outcome resets the streak.
        let moved = RunOutcome {
            progress: true,
            fingerprint: "mutated".into(),
        };
        assert!(store.record_outcome("loop", &moved, 2).unwrap().is_none());
        assert_eq!(store.get("loop").unwrap().state.no_progress_streak, 0);
        for _ in 0..9 {
            store.record_outcome("loop", &stuck, 3).unwrap();
        }
        let other = RunOutcome {
            progress: false,
            fingerprint: "failed: timeout".into(),
        };
        assert!(store.record_outcome("loop", &other, 4).unwrap().is_none());
        for _ in 0..9 {
            assert!(store.record_outcome("loop", &stuck, 5).unwrap().is_none());
        }
        let pause = store
            .record_outcome("loop", &stuck, 6)
            .unwrap()
            .expect("the tenth identical outcome trips the wire");
        assert_eq!(
            pause.reason,
            PauseReason::NoProgressTripWire {
                repeats: NO_PROGRESS_TRIP_WIRE,
                outcome: "failed: provider 429".into(),
            }
        );
        let stored = store.get("loop").unwrap();
        assert!(!stored.enabled);
        assert!(store.poll_due(10_000).unwrap().claims.is_empty());
        // The operator's enable clears the pause and the streak.
        store.set_enabled("loop", true).unwrap();
        let resumed = store.get("loop").unwrap();
        assert!(resumed.state.pause.is_none());
        assert_eq!(resumed.state.no_progress_streak, 0);
    }
}
