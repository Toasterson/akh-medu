//! The Chief-of-Staff log: announcements plus the write-ahead spend ledger.
//!
//! One JSONL file holds three record kinds, each tagged by `contract`:
//!
//! - `lilith/cos-spend-reserved@1`, written **before** a spending call runs;
//! - `lilith/cos-spend-settled@1`, written after it ends (`spent` says whether
//!   the reservation stands);
//! - `lilith/cos-announcement@1`, the operator-facing announcement. The relay
//!   to the operator's CoS thread posts these lines and skips the others.
//!
//! On restart the ledger is rebuilt from the file. A reservation without a
//! settlement counts as spent, because the call may have run before the
//! process died. Each record is appended with a single `write_all` of the
//! whole line to an `O_APPEND` file, so concurrent appends never interleave. A
//! final line cut short by a crash is skipped with a warning and cut off the
//! file. The guard does not fail closed over it, and the next append starts
//! on a fresh line.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::contracts::{CosAnnouncement, Money, RuleCheck, StandingRule, UnixSeconds};
use super::error::{CosError, CosResult};

/// A spend reserved before the call runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpendReservation {
    /// Unique across restarts (UUID v7).
    pub reservation: String,
    /// The tool that will spend.
    pub tool: String,
    /// The amount reserved.
    pub amount: Money,
    /// When it was reserved.
    pub reserved_at: UnixSeconds,
}

/// How a reservation ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpendSettlement {
    /// The reservation this settles.
    pub reservation: String,
    /// Whether the reservation stands (false only when the call certainly
    /// failed).
    pub spent: bool,
    /// When it settled.
    pub settled_at: UnixSeconds,
}

/// One line of the Chief-of-Staff log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "contract")]
pub enum CosLogRecord {
    /// Operator-facing announcement.
    #[serde(rename = "lilith/cos-announcement@1")]
    Announcement(Box<CosAnnouncement>),
    /// Write-ahead spend reservation.
    #[serde(rename = "lilith/cos-spend-reserved@1")]
    SpendReserved(SpendReservation),
    /// Spend settlement.
    #[serde(rename = "lilith/cos-spend-settled@1")]
    SpendSettled(SpendSettlement),
}

/// Where Chief-of-Staff records go.
pub trait CosLog: Send + Sync {
    /// A short description for diagnostics.
    fn describe(&self) -> String;

    /// Durably append one record.
    fn append(&self, record: &CosLogRecord) -> CosResult<()>;
}

/// The JSONL file log.
#[derive(Debug)]
pub struct JsonlLog {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl JsonlLog {
    /// Open (or prepare to create) the log at `path`, returning it with every
    /// record already in it. A truncated final line is skipped with a warning
    /// and cut off, so the next append starts on a fresh line. Any other bad
    /// line is an error: the budget cannot be rebuilt past it.
    pub fn open(path: impl Into<PathBuf>) -> CosResult<(Self, Vec<CosLogRecord>)> {
        let path = path.into();
        let records = Self::load(&path)?;
        Ok((
            Self {
                path,
                write_lock: Mutex::new(()),
            },
            records,
        ))
    }

    fn load(path: &Path) -> CosResult<Vec<CosLogRecord>> {
        let log_error = |line: usize, message: String| CosError::AnnouncementLog {
            path: path.display().to_string(),
            line,
            message,
        };
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
        {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(log_error(0, e.to_string())),
        };
        let mut text = String::new();
        (&file)
            .read_to_string(&mut text)
            .map_err(|e| log_error(0, e.to_string()))?;
        let terminated = text.is_empty() || text.ends_with('\n');
        let lines: Vec<&str> = text.lines().collect();
        let mut records = Vec::with_capacity(lines.len());
        for (index, line) in lines.iter().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<CosLogRecord>(line) {
                Ok(record) => records.push(record),
                Err(e) if index + 1 == lines.len() && !terminated => {
                    tracing::warn!(
                        log = %path.display(),
                        line = index + 1,
                        error = %e,
                        "skipping truncated final line of the Chief-of-Staff log"
                    );
                }
                Err(e) => return Err(log_error(index + 1, e.to_string())),
            }
        }
        if !terminated {
            // Drop the partial tail so the next append starts a fresh line and
            // the fragment never becomes a damaged line in the middle. It was
            // never a complete record: its writer died mid-append, before the
            // call it would have covered could run.
            let keep = text.rfind('\n').map_or(0, |at| at + 1);
            file.set_len(keep as u64)
                .map_err(|e| log_error(lines.len(), e.to_string()))?;
        }
        Ok(records)
    }

    /// Every record in the log (re-read from disk).
    pub fn read_all(&self) -> CosResult<Vec<CosLogRecord>> {
        Self::load(&self.path)
    }

    /// The log's path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl CosLog for JsonlLog {
    fn describe(&self) -> String {
        self.path.display().to_string()
    }

    fn append(&self, record: &CosLogRecord) -> CosResult<()> {
        let failed = |message: String| CosError::LogAppend {
            log: self.describe(),
            message,
        };
        let mut line = serde_json::to_string(record).map_err(|e| failed(e.to_string()))?;
        line.push('\n');
        let _held = self.write_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| failed(e.to_string()))?;
        // One write of the whole line to an O_APPEND file: another writer's
        // record can never land in the middle of this one.
        file.write_all(line.as_bytes())
            .and_then(|()| file.sync_data())
            .map_err(|e| failed(e.to_string()))
    }
}

/// Keeps records in memory (tests, dojo replay).
#[derive(Debug, Default)]
pub struct MemoryLog {
    records: Mutex<Vec<CosLogRecord>>,
}

impl MemoryLog {
    /// Every record so far.
    pub fn records(&self) -> Vec<CosLogRecord> {
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The announcements so far.
    pub fn announcements(&self) -> Vec<CosAnnouncement> {
        announcements(&self.records())
    }
}

impl CosLog for MemoryLog {
    fn describe(&self) -> String {
        "memory".into()
    }

    fn append(&self, record: &CosLogRecord) -> CosResult<()> {
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(record.clone());
        Ok(())
    }
}

/// The announcements among `records`.
pub fn announcements(records: &[CosLogRecord]) -> Vec<CosAnnouncement> {
    records
        .iter()
        .filter_map(|record| match record {
            CosLogRecord::Announcement(a) => Some((**a).clone()),
            _ => None,
        })
        .collect()
}

// ── Spend ledger ────────────────────────────────────────────────────────────

/// The running spend against the configured budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpendLedger {
    limit: Money,
    spent_minor_units: u64,
}

impl SpendLedger {
    /// A fresh ledger with nothing spent.
    pub fn new(limit: Money) -> Self {
        Self {
            limit,
            spent_minor_units: 0,
        }
    }

    /// Rebuild the ledger from the log: every reservation in the budget's
    /// currency counts unless a settlement says it was not spent. An
    /// unsettled reservation counts: the call may have run before a crash.
    pub fn replay(limit: Money, records: &[CosLogRecord]) -> Self {
        let mut reserved: BTreeMap<&str, u64> = BTreeMap::new();
        for record in records {
            match record {
                CosLogRecord::SpendReserved(r) if r.amount.currency == limit.currency => {
                    reserved.insert(&r.reservation, r.amount.minor_units);
                }
                CosLogRecord::SpendSettled(s) if !s.spent => {
                    reserved.remove(s.reservation.as_str());
                }
                _ => {}
            }
        }
        Self {
            spent_minor_units: reserved.values().fold(0u64, |a, b| a.saturating_add(*b)),
            limit,
        }
    }

    /// Minor units still available.
    pub fn remaining(&self) -> u64 {
        self.limit
            .minor_units
            .saturating_sub(self.spent_minor_units)
    }

    /// Reserve `amount`, or refuse if it does not fit.
    pub fn reserve(&mut self, amount: &Money) -> CosResult<RuleCheck> {
        if amount.currency != self.limit.currency {
            return Err(CosError::CurrencyMismatch {
                requested: amount.currency.as_str().to_string(),
                budget: self.limit.currency.as_str().to_string(),
            });
        }
        let remaining = self.remaining();
        if amount.minor_units > remaining {
            return Err(CosError::BudgetExceeded {
                requested: amount.minor_units,
                remaining,
                limit: self.limit.minor_units,
                currency: self.limit.currency.as_str().to_string(),
            });
        }
        self.spent_minor_units += amount.minor_units;
        Ok(RuleCheck {
            rule: StandingRule::WithinBudget,
            evidence: format!(
                "reserved {} {}; {} of {} remaining",
                amount.minor_units,
                amount.currency.as_str(),
                self.remaining(),
                self.limit.minor_units
            ),
        })
    }

    /// Return a reservation whose action certainly did not take effect.
    pub fn release(&mut self, amount: &Money) {
        if amount.currency == self.limit.currency {
            self.spent_minor_units = self.spent_minor_units.saturating_sub(amount.minor_units);
        }
    }
}
