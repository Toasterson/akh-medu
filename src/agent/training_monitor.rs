//! Deterministic monitoring for BabyLLM and LoRA training runs.
//!
//! Trainers publish an atomically-replaced JSON status file. This monitor
//! compares it with a durable per-run watermark, emits only meaningful
//! transitions, and can request one narrowly-scoped control operation:
//! `checkpoint_and_stop`. Model reasoning may explain an event, but is never
//! the safety controller.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::error::{AgentError, AgentResult};
use crate::engine::Engine;

const MONITOR_PREFIX: &str = "training-monitor:";

fn monitor_error(message: impl Into<String>) -> AgentError {
    AgentError::ToolExecution {
        tool_name: "training_monitor".into(),
        message: message.into(),
    }
}

/// Atomic status contract written by a trainer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingStatus {
    pub run_id: String,
    pub state: String,
    pub updated_at: u64,
    /// Last observed trainer progress (step/state/checkpoint), distinct from
    /// the process heartbeat in `updated_at`.
    #[serde(default)]
    pub progress_updated_at: Option<u64>,
    #[serde(default)]
    pub step: u64,
    #[serde(default)]
    pub total_steps: Option<u64>,
    #[serde(default)]
    pub epoch: Option<f64>,
    #[serde(default)]
    pub loss: Option<f64>,
    #[serde(default)]
    pub eval_loss: Option<f64>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub checkpoint_path: Option<String>,
    #[serde(default)]
    pub disk_free_bytes: Option<u64>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, serde_json::Value>,
}

/// Durable comparison watermark. It deliberately stores only monitor state,
/// never credentials or the full training corpus.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
struct MonitorWatermark {
    last_state: String,
    last_step: u64,
    last_updated_at: u64,
    last_progress_updated_at: u64,
    last_reported_bucket: u64,
    last_checkpoint_mtime_ns: u64,
    stalled_reported: bool,
    control_requested: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingControl {
    CheckpointAndStop,
}

/// Change-only result returned to the reactor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingObservation {
    pub changed: bool,
    pub severity: TrainingSeverity,
    pub summary: String,
    pub status: TrainingStatus,
    pub recommended_control: Option<TrainingControl>,
}

#[derive(Debug, Clone)]
pub struct TrainingMonitorConfig {
    pub run_id: String,
    pub status_path: PathBuf,
    pub control_path: Option<PathBuf>,
    pub stall_after_seconds: u64,
    pub progress_report_every_steps: u64,
    pub min_disk_free_bytes: u64,
    pub checkpoint_and_stop_on_stall: bool,
}

impl TrainingMonitorConfig {
    pub fn new(run_id: impl Into<String>, status_path: impl Into<PathBuf>) -> Self {
        Self {
            run_id: run_id.into(),
            status_path: status_path.into(),
            control_path: None,
            stall_after_seconds: 300,
            progress_report_every_steps: 100,
            min_disk_free_bytes: 5 * 1024 * 1024 * 1024,
            checkpoint_and_stop_on_stall: false,
        }
    }
}

/// Read, compare, and persist one deterministic monitor observation.
pub fn inspect_training(
    engine: &Engine,
    config: &TrainingMonitorConfig,
    now: u64,
) -> AgentResult<TrainingObservation> {
    if engine.store().durable.is_none() {
        return Err(monitor_error(
            "training monitor requires an engine data directory for durable watermarks",
        ));
    }
    let bytes = std::fs::read(&config.status_path).map_err(|error| {
        monitor_error(format!(
            "read status {}: {error}",
            config.status_path.display()
        ))
    })?;
    let status: TrainingStatus = serde_json::from_slice(&bytes)
        .map_err(|error| monitor_error(format!("parse training status: {error}")))?;
    validate_status(&status, &config.run_id)?;

    let key = format!("{MONITOR_PREFIX}{}", config.run_id);
    let previous = engine
        .store()
        .get_meta(key.as_bytes())
        .map_err(|error| AgentError::Engine(Box::new(error.into())))?
        .and_then(|bytes| bincode::deserialize::<MonitorWatermark>(&bytes).ok())
        .unwrap_or_default();

    let bucket_size = config.progress_report_every_steps.max(1);
    let bucket = status.step / bucket_size;
    let progress_updated_at = status.progress_updated_at.unwrap_or(status.updated_at);
    let stalled = !is_terminal(&status.state)
        && now.saturating_sub(progress_updated_at) > config.stall_after_seconds;
    let regressed = status.step < previous.last_step
        || status.updated_at < previous.last_updated_at
        || progress_updated_at < previous.last_progress_updated_at;
    let disk_critical = status
        .disk_free_bytes
        .is_some_and(|free| free < config.min_disk_free_bytes);
    let checkpoint_hazard = status.state.eq_ignore_ascii_case("completed")
        && status
            .checkpoint_path
            .as_ref()
            .is_none_or(|path| !Path::new(path).exists());
    let state_changed = previous.last_state.is_empty() || previous.last_state != status.state;
    let checkpoint_mtime_ns = status
        .metrics
        .get("checkpoint_mtime_ns")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    let checkpoint_changed = checkpoint_mtime_ns > previous.last_checkpoint_mtime_ns;
    let progress_changed = bucket > previous.last_reported_bucket || checkpoint_changed;

    let (changed, severity, summary, recommended_control) = if regressed {
        (
            true,
            TrainingSeverity::Critical,
            format!(
                "training {} telemetry regressed: step {} after {}, updated_at {} after {}, progress_updated_at {} after {}",
                status.run_id,
                status.step,
                previous.last_step,
                status.updated_at,
                previous.last_updated_at,
                progress_updated_at,
                previous.last_progress_updated_at
            ),
            Some(TrainingControl::CheckpointAndStop),
        )
    } else if disk_critical {
        (
            true,
            TrainingSeverity::Critical,
            format!(
                "training {} disk guard tripped at step {}: {} bytes free",
                status.run_id,
                status.step,
                status.disk_free_bytes.unwrap_or_default()
            ),
            Some(TrainingControl::CheckpointAndStop),
        )
    } else if checkpoint_hazard {
        (
            true,
            TrainingSeverity::Warning,
            format!(
                "training {} reported completion without a readable checkpoint",
                status.run_id
            ),
            None,
        )
    } else if stalled && !previous.stalled_reported {
        (
            true,
            TrainingSeverity::Warning,
            format!(
                "training {} progress stalled for {} seconds at step {} (heartbeat may still be live)",
                status.run_id,
                now.saturating_sub(progress_updated_at),
                status.step
            ),
            config
                .checkpoint_and_stop_on_stall
                .then_some(TrainingControl::CheckpointAndStop),
        )
    } else if state_changed {
        (
            true,
            if status.state.eq_ignore_ascii_case("failed") {
                TrainingSeverity::Critical
            } else {
                TrainingSeverity::Info
            },
            format!(
                "training {} entered state {} at step {}{}",
                status.run_id,
                status.state,
                status.step,
                status
                    .total_steps
                    .map(|total| format!("/{total}"))
                    .unwrap_or_default()
            ),
            None,
        )
    } else if progress_changed {
        (
            true,
            TrainingSeverity::Info,
            if checkpoint_changed && bucket <= previous.last_reported_bucket {
                format!(
                    "training {} wrote a new durable checkpoint at step {}",
                    status.run_id, status.step
                )
            } else {
                format!(
                    "training {} progressed to step {}{}{}",
                    status.run_id,
                    status.step,
                    status
                        .total_steps
                        .map(|total| format!("/{total}"))
                        .unwrap_or_default(),
                    status
                        .loss
                        .map(|loss| format!(", loss={loss:.6}"))
                        .unwrap_or_default()
                )
            },
            None,
        )
    } else {
        (
            false,
            TrainingSeverity::Info,
            format!(
                "training {} unchanged at step {}",
                status.run_id, status.step
            ),
            None,
        )
    };

    let mut watermark = MonitorWatermark {
        last_state: status.state.clone(),
        last_step: status.step,
        last_updated_at: status.updated_at,
        last_progress_updated_at: progress_updated_at,
        last_reported_bucket: previous.last_reported_bucket.max(bucket),
        last_checkpoint_mtime_ns: previous.last_checkpoint_mtime_ns.max(checkpoint_mtime_ns),
        stalled_reported: stalled,
        control_requested: previous.control_requested,
    };
    if let Some(control) = recommended_control
        && let Some(path) = &config.control_path
        && !previous.control_requested
    {
        write_control_request(path, &status.run_id, control, now, &summary)?;
        watermark.control_requested = true;
    }
    let encoded = bincode::serialize(&watermark)
        .map_err(|error| monitor_error(format!("serialize monitor watermark: {error}")))?;
    engine
        .store()
        .put_meta(key.as_bytes(), &encoded)
        .map_err(|error| AgentError::Engine(Box::new(error.into())))?;

    Ok(TrainingObservation {
        changed,
        severity,
        summary,
        status,
        recommended_control,
    })
}

fn validate_status(status: &TrainingStatus, expected_run_id: &str) -> AgentResult<()> {
    if status.run_id != expected_run_id {
        return Err(monitor_error(format!(
            "status run_id {:?} does not match configured run {:?}",
            status.run_id, expected_run_id
        )));
    }
    if status.run_id.trim().is_empty() || status.state.trim().is_empty() {
        return Err(monitor_error("run_id and state may not be empty"));
    }
    for (name, value) in [
        ("epoch", status.epoch),
        ("loss", status.loss),
        ("eval_loss", status.eval_loss),
        ("learning_rate", status.learning_rate),
    ] {
        if value.is_some_and(|number| !number.is_finite()) {
            return Err(monitor_error(format!("non-finite {name} metric")));
        }
    }
    Ok(())
}

fn is_terminal(state: &str) -> bool {
    state.eq_ignore_ascii_case("completed")
        || state.eq_ignore_ascii_case("failed")
        || state.eq_ignore_ascii_case("stopped")
}

fn write_control_request(
    path: &Path,
    run_id: &str,
    control: TrainingControl,
    now: u64,
    reason: &str,
) -> AgentResult<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            monitor_error(format!(
                "create control directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let request = serde_json::json!({
        "run_id": run_id,
        "command": match control { TrainingControl::CheckpointAndStop => "checkpoint_and_stop" },
        "requested_at": now,
        "reason": reason,
    });
    let bytes = serde_json::to_vec_pretty(&request)
        .map_err(|error| monitor_error(format!("serialize control request: {error}")))?;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    std::fs::write(&tmp, bytes).map_err(|error| {
        monitor_error(format!("write control request {}: {error}", tmp.display()))
    })?;
    std::fs::rename(&tmp, path).map_err(|error| {
        monitor_error(format!(
            "publish control request {} -> {}: {error}",
            tmp.display(),
            path.display()
        ))
    })
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

    fn write_status(path: &Path, step: u64, updated_at: u64, disk: u64) {
        let status = TrainingStatus {
            run_id: "run-1".into(),
            state: "training".into(),
            updated_at,
            progress_updated_at: Some(updated_at),
            step,
            total_steps: Some(1000),
            epoch: Some(0.5),
            loss: Some(1.25),
            eval_loss: None,
            learning_rate: Some(0.0002),
            checkpoint_path: None,
            disk_free_bytes: Some(disk),
            message: None,
            metrics: BTreeMap::new(),
        };
        std::fs::write(path, serde_json::to_vec(&status).unwrap()).unwrap();
    }

    #[test]
    fn reports_changes_once_and_keeps_unchanged_ticks_quiet() {
        let temp = tempfile::tempdir().unwrap();
        let status_path = temp.path().join("status.json");
        write_status(&status_path, 10, 100, 20 * 1024 * 1024 * 1024);
        let config = TrainingMonitorConfig::new("run-1", &status_path);
        let engine = engine();
        assert!(inspect_training(&engine, &config, 101).unwrap().changed);
        assert!(!inspect_training(&engine, &config, 102).unwrap().changed);
    }

    #[test]
    fn disk_guard_writes_only_checkpoint_and_stop() {
        let temp = tempfile::tempdir().unwrap();
        let status_path = temp.path().join("status.json");
        let control_path = temp.path().join("control.json");
        write_status(&status_path, 10, 100, 128);
        let mut config = TrainingMonitorConfig::new("run-1", &status_path);
        config.control_path = Some(control_path.clone());
        let observation = inspect_training(&engine(), &config, 101).unwrap();
        assert_eq!(
            observation.recommended_control,
            Some(TrainingControl::CheckpointAndStop)
        );
        let request: serde_json::Value =
            serde_json::from_slice(&std::fs::read(control_path).unwrap()).unwrap();
        assert_eq!(request["command"], "checkpoint_and_stop");
    }

    #[test]
    fn telemetry_regression_is_fail_closed() {
        let temp = tempfile::tempdir().unwrap();
        let status_path = temp.path().join("status.json");
        let config = TrainingMonitorConfig::new("run-1", &status_path);
        let engine = engine();
        write_status(&status_path, 100, 100, 20 * 1024 * 1024 * 1024);
        inspect_training(&engine, &config, 101).unwrap();
        write_status(&status_path, 90, 102, 20 * 1024 * 1024 * 1024);
        let observation = inspect_training(&engine, &config, 103).unwrap();
        assert_eq!(observation.severity, TrainingSeverity::Critical);
    }

    #[test]
    fn live_heartbeat_does_not_hide_stalled_training_progress() {
        let temp = tempfile::tempdir().unwrap();
        let status_path = temp.path().join("status.json");
        write_status(&status_path, 10, 1_000, 20 * 1024 * 1024 * 1024);
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&status_path).unwrap()).unwrap();
        value["progress_updated_at"] = serde_json::json!(100);
        std::fs::write(&status_path, serde_json::to_vec(&value).unwrap()).unwrap();
        let mut config = TrainingMonitorConfig::new("run-1", &status_path);
        config.stall_after_seconds = 300;

        let observation = inspect_training(&engine(), &config, 1_001).unwrap();
        assert!(observation.changed);
        assert_eq!(observation.severity, TrainingSeverity::Warning);
        assert!(observation.summary.contains("progress stalled"));
    }
}
