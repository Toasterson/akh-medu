//! The Chief-of-Staff tool-call guard: classifies every call by the profile,
//! enforces the standing rules before an operator-class action runs, and
//! announces the action afterwards with its receipts, reason and rollback
//! path (ADR 116 Decision 1, revision 4).

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::agent::permission::{GuardTicket, ToolCallGuard};
use crate::agent::tool::ToolInput;
use crate::agent::tool_manifest::ToolManifest;

use super::contracts::{
    ActionOutcome, CosAnnouncement, CosDocument, Money, OperatorAction, Receipt, Receipts,
    RollbackPath, RuleCheck, StandingRule, UnixSeconds,
};
use super::error::{CosError, CosResult};
use super::policy::{
    AncestryOracle, AutonomyTier, ChiefOfStaffConfig, CosPermissionProfile, DECLARATION_PARAM,
    Declaration, GitAncestry, SpendLedger, TierGrant, check_main_only, check_verified_deletion,
};

// ── Announcement sinks ──────────────────────────────────────────────────────

/// Where operator-class announcements go (the operator's CoS thread relay, or
/// the akh's configured sink).
pub trait AnnouncementSink: Send + Sync {
    /// A short description for diagnostics.
    fn describe(&self) -> String;

    /// Deliver one announcement.
    fn announce(&self, announcement: &CosAnnouncement) -> CosResult<()>;
}

/// Appends each announcement as one `lilith/cos-announcement@1` JSON line.
/// A relay posts new lines to the operator's CoS thread; the file is also the
/// durable record the spend ledger is rebuilt from.
#[derive(Debug)]
pub struct JsonlAnnouncementSink {
    path: PathBuf,
    write_lock: Mutex<()>,
}

impl JsonlAnnouncementSink {
    /// A sink appending to `path` (created on first announcement).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            write_lock: Mutex::new(()),
        }
    }

    /// Every announcement in the log, in order. A missing file is empty.
    pub fn read_all(&self) -> CosResult<Vec<CosAnnouncement>> {
        read_announcements(&self.path)
    }
}

fn read_announcements(path: &Path) -> CosResult<Vec<CosAnnouncement>> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(CosError::AnnouncementLog {
                path: path.display().to_string(),
                line: 0,
                message: e.to_string(),
            });
        }
    };
    let log_error = |line: usize, message: String| CosError::AnnouncementLog {
        path: path.display().to_string(),
        line,
        message,
    };
    let mut announcements = Vec::new();
    for (index, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| log_error(index + 1, e.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<CosDocument>(&line) {
            Ok(CosDocument::Announcement(announcement)) => announcements.push(announcement),
            Ok(other) => {
                return Err(log_error(
                    index + 1,
                    format!("expected an announcement, found {}", other.contract_id()),
                ));
            }
            Err(e) => return Err(log_error(index + 1, e.to_string())),
        }
    }
    Ok(announcements)
}

impl AnnouncementSink for JsonlAnnouncementSink {
    fn describe(&self) -> String {
        self.path.display().to_string()
    }

    fn announce(&self, announcement: &CosAnnouncement) -> CosResult<()> {
        let failed = |message: String| CosError::AnnouncementFailed {
            sink: self.describe(),
            message,
        };
        let line = serde_json::to_string(&CosDocument::Announcement(announcement.clone()))
            .map_err(|e| failed(e.to_string()))?;
        let _held = self.write_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| failed(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| failed(e.to_string()))?;
        file.sync_data().map_err(|e| failed(e.to_string()))
    }
}

/// Keeps announcements in memory (tests, dojo replay).
#[derive(Debug, Default)]
pub struct MemoryAnnouncementSink {
    announcements: Mutex<Vec<CosAnnouncement>>,
}

impl MemoryAnnouncementSink {
    /// Everything announced so far.
    pub fn announcements(&self) -> Vec<CosAnnouncement> {
        self.announcements
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl AnnouncementSink for MemoryAnnouncementSink {
    fn describe(&self) -> String {
        "memory".into()
    }

    fn announce(&self, announcement: &CosAnnouncement) -> CosResult<()> {
        self.announcements
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(announcement.clone());
        Ok(())
    }
}

// ── Admission and settlement ────────────────────────────────────────────────

/// An operator-class action that passed every standing-rule check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedAction {
    /// The tool carrying it.
    pub tool: String,
    /// The action.
    pub action: OperatorAction,
    /// Her receipts plus the runtime's own rule receipts.
    pub receipts: Receipts,
    /// Why now.
    pub reason: String,
    /// How to undo it.
    pub rollback: RollbackPath,
    /// The checks that passed.
    pub rule_checks: Vec<RuleCheck>,
}

/// What the guard admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// T0/T1: autonomous, nothing to announce.
    Autonomous { tool: String, tier: AutonomyTier },
    /// T2: autonomous, carries receipts.
    Coordination { tool: String, receipts: Receipts },
    /// T3: autonomous, announced after it runs.
    OperatorClass(Box<AuthorizedAction>),
}

/// What settling an admission produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Settlement {
    /// T0/T1: nothing further.
    Recorded { tool: String, tier: AutonomyTier },
    /// T2: the post went out with these receipts.
    Coordinated { tool: String, receipts: Receipts },
    /// T3: this announcement was delivered.
    Announced(Box<CosAnnouncement>),
}

/// Lilith's Chief-of-Staff guard.
pub struct CosGuard {
    profile: CosPermissionProfile,
    ancestry: Box<dyn AncestryOracle>,
    sink: Arc<dyn AnnouncementSink>,
    ledger: Mutex<SpendLedger>,
    pending: Mutex<HashMap<u64, Admission>>,
    next_ticket: AtomicU64,
    clock: fn() -> UnixSeconds,
}

impl std::fmt::Debug for CosGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CosGuard")
            .field("akh", &self.profile.akh)
            .field("sink", &self.sink.describe())
            .finish_non_exhaustive()
    }
}

fn system_clock() -> UnixSeconds {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl CosGuard {
    /// A guard over `profile` with a spend `ledger`, an ancestry oracle and an
    /// announcement sink.
    pub fn new(
        profile: CosPermissionProfile,
        ledger: SpendLedger,
        ancestry: Box<dyn AncestryOracle>,
        sink: Arc<dyn AnnouncementSink>,
    ) -> Self {
        Self {
            profile,
            ancestry,
            sink,
            ledger: Mutex::new(ledger),
            pending: Mutex::new(HashMap::new()),
            next_ticket: AtomicU64::new(1),
            clock: system_clock,
        }
    }

    /// Replace the clock (deterministic tests and replay).
    pub fn with_clock(mut self, clock: fn() -> UnixSeconds) -> Self {
        self.clock = clock;
        self
    }

    /// Build the guard from an akh's `[chief_of_staff]` table: load the
    /// profile, check ancestry with git in `repository`, announce to the JSONL
    /// log, and rebuild the spend ledger from that log.
    pub fn from_config(config: &ChiefOfStaffConfig) -> CosResult<Self> {
        let profile = CosPermissionProfile::load(&config.profile)?;
        let sink = JsonlAnnouncementSink::new(&config.announcements);
        let ledger = SpendLedger::replay(config.budget.clone(), &sink.read_all()?);
        Ok(Self::new(
            profile,
            ledger,
            Box::new(GitAncestry::new(&config.repository)),
            Arc::new(sink),
        ))
    }

    /// The profile in force.
    pub fn profile(&self) -> &CosPermissionProfile {
        &self.profile
    }

    /// Minor units of budget left.
    pub fn remaining_budget(&self) -> u64 {
        self.ledger
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remaining()
    }

    /// Classify and check one call. Refusals carry the caller's options.
    pub fn admit(
        &self,
        manifest: &ToolManifest,
        declaration: Option<Declaration>,
    ) -> CosResult<Admission> {
        let tool = manifest.name.clone();
        let class_tier =
            self.profile
                .classify(manifest)
                .tier()
                .ok_or_else(|| CosError::OutsideGrant {
                    tool: tool.clone(),
                    what: tool.clone(),
                })?;
        if let Some(declared) = declaration.as_ref().map(Declaration::tier)
            && declared < class_tier
        {
            return Err(CosError::DeclarationBelowTier {
                tool,
                class: class_tier,
                declared,
            });
        }
        let tier = declaration
            .as_ref()
            .map_or(class_tier, |d| d.tier().max(class_tier));
        if self.profile.grant(tier) == TierGrant::Gated {
            return Err(CosError::TierGated { tool, tier });
        }
        match (tier, declaration) {
            (AutonomyTier::Read | AutonomyTier::OwnState, _) => {
                Ok(Admission::Autonomous { tool, tier })
            }
            (_, None) => Err(CosError::MissingDeclaration { tool, tier }),
            (_, Some(Declaration::Coordination { receipts })) => Ok(Admission::Coordination {
                receipts: require_receipts(&tool, receipts)?,
                tool,
            }),
            (
                _,
                Some(Declaration::OperatorClass {
                    action,
                    receipts,
                    reason,
                    rollback,
                }),
            ) => self
                .authorize(tool, action, receipts, reason, rollback)
                .map(|authorized| Admission::OperatorClass(Box::new(authorized))),
        }
    }

    fn authorize(
        &self,
        tool: String,
        action: OperatorAction,
        receipts: Vec<Receipt>,
        reason: String,
        rollback: RollbackPath,
    ) -> CosResult<AuthorizedAction> {
        let mut receipts = require_receipts(&tool, receipts)?;
        if reason.trim().is_empty() {
            return Err(CosError::MissingReason { tool });
        }
        let now = (self.clock)();
        let mut rule_checks = vec![RuleCheck {
            rule: StandingRule::ReceiptsPresent,
            evidence: format!("{} receipt(s)", receipts.as_slice().len()),
        }];
        match &action {
            OperatorAction::ActivateAdapter { adapter } => {
                return Err(CosError::OutsideGrant {
                    tool,
                    what: format!("activate-adapter {adapter}"),
                });
            }
            OperatorAction::Deploy { artifact_sha, .. }
            | OperatorAction::Restart { artifact_sha, .. } => {
                let (check, receipt) = check_main_only(
                    self.ancestry.as_ref(),
                    artifact_sha,
                    &self.profile.main_ref,
                    now,
                )?;
                rule_checks.push(check);
                receipts.push(receipt);
            }
            OperatorAction::Spend { amount, .. } => rule_checks.push(
                self.ledger
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .reserve(amount)?,
            ),
            OperatorAction::Delete {
                target,
                reversibility,
                copy,
            } => rule_checks.push(check_verified_deletion(
                "delete",
                target,
                *reversibility,
                copy.as_ref(),
            )?),
            OperatorAction::Overwrite {
                target,
                reversibility,
                copy,
            } => rule_checks.push(check_verified_deletion(
                "overwrite",
                target,
                *reversibility,
                copy.as_ref(),
            )?),
            OperatorAction::Merge { .. }
            | OperatorAction::Tracker { .. }
            | OperatorAction::Outward { .. } => {}
        }
        Ok(AuthorizedAction {
            tool,
            action,
            receipts,
            reason,
            rollback,
            rule_checks,
        })
    }

    /// Settle an admission once its call ended. For an operator-class action
    /// this delivers the announcement; a failed spend returns its reservation.
    pub fn settle(&self, admission: Admission, outcome: ActionOutcome) -> CosResult<Settlement> {
        match admission {
            Admission::Autonomous { tool, tier } => Ok(Settlement::Recorded { tool, tier }),
            Admission::Coordination { tool, receipts } => {
                Ok(Settlement::Coordinated { tool, receipts })
            }
            Admission::OperatorClass(authorized) => {
                if let OperatorAction::Spend { amount, .. } = &authorized.action
                    && !outcome.succeeded()
                {
                    self.release(amount);
                }
                let announcement = CosAnnouncement {
                    akh: self.profile.akh.clone(),
                    announced_at: (self.clock)(),
                    tool: authorized.tool,
                    action: authorized.action,
                    receipts: authorized.receipts,
                    reason: authorized.reason,
                    rollback: authorized.rollback,
                    rule_checks: authorized.rule_checks,
                    outcome,
                };
                self.sink.announce(&announcement)?;
                Ok(Settlement::Announced(Box::new(announcement)))
            }
        }
    }

    fn release(&self, amount: &Money) {
        self.ledger
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release(amount);
    }
}

fn require_receipts(tool: &str, receipts: Vec<Receipt>) -> CosResult<Receipts> {
    Receipts::try_from(receipts).map_err(|_| CosError::MissingReceipts {
        tool: tool.to_string(),
    })
}

/// Parse the `cos_declaration` argument. Models sometimes send nested JSON as
/// a string, so a string holding JSON is accepted too.
fn parse_declaration(tool: &str, raw: serde_json::Value) -> CosResult<Declaration> {
    let malformed = |message: String| CosError::MalformedDeclaration {
        tool: tool.to_string(),
        message,
    };
    match raw {
        serde_json::Value::String(text) => {
            serde_json::from_str(&text).map_err(|e| malformed(e.to_string()))
        }
        value => serde_json::from_value(value).map_err(|e| malformed(e.to_string())),
    }
}

impl ToolCallGuard for CosGuard {
    fn admit(&self, manifest: &ToolManifest, input: &mut ToolInput) -> Result<GuardTicket, String> {
        let admission = input
            .params
            .remove(DECLARATION_PARAM)
            .map(|raw| parse_declaration(&manifest.name, raw))
            .transpose()
            .and_then(|declaration| CosGuard::admit(self, manifest, declaration))
            .map_err(|e| e.to_tool_refusal())?;
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(ticket, admission);
        Ok(GuardTicket(ticket))
    }

    fn settle(&self, ticket: GuardTicket, succeeded: bool, detail: &str) -> Result<(), String> {
        let admission = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&ticket.0)
            .ok_or(CosError::UnknownTicket { ticket: ticket.0 })
            .map_err(|e| e.to_tool_refusal())?;
        let detail = detail.to_string();
        let outcome = if succeeded {
            ActionOutcome::Succeeded { detail }
        } else {
            ActionOutcome::Failed { detail }
        };
        CosGuard::settle(self, admission, outcome)
            .map(|_| ())
            .map_err(|e| e.to_tool_refusal())
    }
}
