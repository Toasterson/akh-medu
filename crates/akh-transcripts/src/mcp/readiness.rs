//! `/readyz` state: "a `SELECT 1` succeeded within the last 30 s" — the
//! contract's own definition, kept as a small shared clock rather than a
//! background poller so `/readyz` always reflects a probe that ran recently
//! against *this* request, not a stale timer tick.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use sea_orm::{ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement};

/// Ready if a `SELECT 1` has succeeded within this window.
const READY_WINDOW: Duration = Duration::from_secs(30);

pub struct Readiness {
    last_ok: Mutex<Option<Instant>>,
}

impl Readiness {
    pub fn new() -> Self {
        Self {
            last_ok: Mutex::new(None),
        }
    }

    /// Run `SELECT 1` against `db`, recording success. Returns whether *this*
    /// probe succeeded (the caller can also consult [`Readiness::is_ready`]
    /// afterwards to apply the 30 s grace window on a failed probe).
    pub async fn probe(&self, db: &DatabaseConnection) -> bool {
        let ok = db
            .execute(Statement::from_string(
                DatabaseBackend::Postgres,
                "SELECT 1".to_string(),
            ))
            .await
            .is_ok();
        if ok {
            let mut guard = self.last_ok.lock().unwrap_or_else(|p| p.into_inner());
            *guard = Some(Instant::now());
        }
        ok
    }

    /// Whether a `SELECT 1` has succeeded within the last 30 s, regardless of
    /// whether the most recent probe itself succeeded.
    pub fn is_ready(&self) -> bool {
        let guard = self.last_ok.lock().unwrap_or_else(|p| p.into_inner());
        guard
            .map(|instant| instant.elapsed() < READY_WINDOW)
            .unwrap_or(false)
    }
}

impl Default for Readiness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_ready_before_any_successful_probe() {
        let readiness = Readiness::new();
        assert!(!readiness.is_ready());
    }

    #[test]
    fn ready_immediately_after_recording_success() {
        let readiness = Readiness::new();
        {
            let mut guard = readiness.last_ok.lock().unwrap();
            *guard = Some(Instant::now());
        }
        assert!(readiness.is_ready());
    }

    #[test]
    fn not_ready_once_the_window_has_elapsed() {
        let readiness = Readiness::new();
        {
            let mut guard = readiness.last_ok.lock().unwrap();
            *guard = Some(Instant::now() - Duration::from_secs(31));
        }
        assert!(!readiness.is_ready());
    }
}
