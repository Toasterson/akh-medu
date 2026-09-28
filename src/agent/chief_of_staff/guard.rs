//! The Chief-of-Staff tool-call guard. It classifies every call by the
//! profile and checks the declaration against the call's real arguments. It
//! enforces the standing rules before an operator-class action runs, and
//! announces the action afterwards with its arguments, receipts, reason and
//! rollback path (ADR 116 Decision 1, revision 4).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use serde_json::{Map, Value};

use crate::agent::permission::{CallOutcome, GuardTicket, ToolCallGuard};
use crate::agent::tool::ToolInput;
use crate::agent::tool_manifest::{ToolManifest, ToolParamSchema};
use crate::fs::LocalFs;

use super::contracts::{
    ActionOutcome, CosAnnouncement, OperatorAction, Receipt, Receipts, RollbackPath, RuleCheck,
    StandingRule, UnixSeconds,
};
use super::error::{CosError, CosResult, DeletionGap};
use super::ledger::{
    CosLog, CosLogRecord, JsonlLog, SpendLedger, SpendReservation, SpendSettlement,
};
use super::policy::{
    AncestryOracle, AutonomyTier, ChiefOfStaffConfig, CosPermissionProfile, DECLARATION_PARAM,
    Declaration, DigestOracle, GitAncestry, LocalDigest, TierGrant, check_argument_refs,
    check_main_only, check_verified_target,
};
use super::redact::redact_arguments;

/// The declaration schema (refs inlined) advertised as the `cos_declaration`
/// parameter, so grammar-constrained backends can emit it.
static DECLARATION_SCHEMA: LazyLock<Value> = LazyLock::new(|| {
    let mut schema: Value = serde_json::from_str(include_str!(
        "../../../skills/chief-of-staff/schemas/cos-declaration.schema.json"
    ))
    .expect("the bundled cos-declaration schema is valid JSON");
    if let Some(map) = schema.as_object_mut() {
        for key in ["$schema", "$id", "title"] {
            map.remove(key);
        }
    }
    schema
});

// ── Admission and settlement ────────────────────────────────────────────────

/// An operator-class action that passed every standing-rule check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedAction {
    /// The tool carrying it.
    pub tool: String,
    /// The tool's real arguments, redacted for announcement.
    pub arguments: Map<String, Value>,
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
    /// The write-ahead spend reservation, for a spend.
    pub reservation: Option<SpendReservation>,
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
    digests: Box<dyn DigestOracle>,
    /// Resolves file_io/edit paths exactly as the tools do (their write roots).
    write_fs: Option<LocalFs>,
    log: Arc<dyn CosLog>,
    ledger: Mutex<SpendLedger>,
    pending: Mutex<HashMap<u64, Admission>>,
    next_ticket: AtomicU64,
    clock: fn() -> UnixSeconds,
}

impl std::fmt::Debug for CosGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CosGuard")
            .field("akh", &self.profile.akh)
            .field("log", &self.log.describe())
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
    /// A guard over `profile` with a spend `ledger`, an ancestry oracle and a
    /// log. Digests default to the local filesystem; no write roots are known
    /// until [`Self::with_write_roots`].
    pub fn new(
        profile: CosPermissionProfile,
        ledger: SpendLedger,
        ancestry: Box<dyn AncestryOracle>,
        log: Arc<dyn CosLog>,
    ) -> Self {
        Self {
            profile,
            ancestry,
            digests: Box::new(LocalDigest),
            write_fs: None,
            log,
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

    /// Replace the digest oracle.
    pub fn with_digests(mut self, digests: Box<dyn DigestOracle>) -> Self {
        self.digests = digests;
        self
    }

    /// The akh's file_io/edit write roots, so overwrite targets resolve the
    /// way the tools resolve them.
    pub fn with_write_roots(mut self, write_roots: Vec<PathBuf>) -> Self {
        self.write_fs = Some(LocalFs::new(Vec::new(), write_roots));
        self
    }

    /// Build the guard from an akh's `[chief_of_staff]` table. It refuses a
    /// log inside `write_roots`, loads the profile, rebuilds the spend ledger
    /// from the log (unsettled reservations count as spent) and checks
    /// ancestry with git in `repository`.
    pub fn from_config(config: &ChiefOfStaffConfig, write_roots: &[PathBuf]) -> CosResult<Self> {
        config.check_log_outside(write_roots)?;
        let profile = CosPermissionProfile::load(&config.profile)?;
        let (log, records) = JsonlLog::open(&config.announcements)?;
        let ledger = SpendLedger::replay(config.budget.clone(), &records);
        Ok(Self::new(
            profile,
            ledger,
            Box::new(GitAncestry::new(&config.repository)),
            Arc::new(log),
        )
        .with_write_roots(write_roots.to_vec()))
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

    /// Admitted calls not yet settled.
    pub fn pending_admissions(&self) -> usize {
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Resolve a path a file-writing tool will write, as the tool does.
    fn resolve_written(&self, raw: &str) -> Option<PathBuf> {
        match &self.write_fs {
            Some(fs) => fs.resolve_write(Path::new(raw)).ok(),
            None => Some(crate::fs::canonicalize_existing_prefix(Path::new(raw))),
        }
    }

    /// Resolve a declared delete/overwrite target or copy location.
    fn resolve_declared(&self, raw: &str) -> PathBuf {
        let path = Path::new(raw);
        if path.is_absolute() {
            return crate::fs::canonicalize_existing_prefix(path);
        }
        self.resolve_written(raw)
            .unwrap_or_else(|| path.to_path_buf())
    }

    /// Classify and check one call against its real `arguments` (declaration
    /// already stripped). Refusals carry the caller's options.
    pub fn admit(
        &self,
        manifest: &ToolManifest,
        declaration: Option<Declaration>,
        arguments: &Map<String, Value>,
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

        // A file_io/edit write to an existing file is an overwrite: it needs
        // an operator-class `overwrite` declaration naming that file.
        let overwrite = self
            .profile
            .overwrite_spec(&tool)
            .and_then(|spec| spec.written_path(arguments))
            .and_then(|raw| self.resolve_written(raw))
            .filter(|path| path.exists());
        let undeclared_overwrite = |path: &Path| CosError::UnverifiedDeletion {
            operation: "overwrite",
            target: path.display().to_string(),
            gap: DeletionGap::UndeclaredOverwrite,
        };
        let required = if overwrite.is_some() {
            AutonomyTier::OperatorClass
        } else {
            class_tier
        };

        if let Some(declared) = declaration.as_ref().map(Declaration::tier)
            && declared < required
        {
            return Err(match &overwrite {
                Some(path) => undeclared_overwrite(path),
                None => CosError::DeclarationBelowTier {
                    tool,
                    class: class_tier,
                    declared,
                },
            });
        }
        let tier = declaration
            .as_ref()
            .map_or(required, |d| d.tier().max(required));
        if self.profile.grant(tier) == TierGrant::Gated {
            return Err(CosError::TierGated { tool, tier });
        }
        if tier >= AutonomyTier::Coordination
            && let Some(marker) = self.profile.adapter_marker_in(arguments)
        {
            return Err(CosError::OutsideGrant {
                tool,
                what: format!("adapter activation (argument marker `{marker}`)"),
            });
        }

        let now = (self.clock)();
        let argument_checks = if self.profile.is_artifact_tool(&tool) {
            check_argument_refs(
                self.ancestry.as_ref(),
                arguments,
                &self.profile.main_ref,
                now,
            )?
        } else {
            Vec::new()
        };

        match (tier, declaration) {
            (AutonomyTier::Read | AutonomyTier::OwnState, _) => {
                Ok(Admission::Autonomous { tool, tier })
            }
            (_, None) => Err(match &overwrite {
                Some(path) => undeclared_overwrite(path),
                None => CosError::MissingDeclaration { tool, tier },
            }),
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
                .authorize(Claim {
                    tool,
                    arguments,
                    action,
                    receipts,
                    reason,
                    rollback,
                    argument_checks,
                    overwrite,
                    now,
                })
                .map(|authorized| Admission::OperatorClass(Box::new(authorized))),
        }
    }

    fn authorize(&self, claim: Claim<'_>) -> CosResult<AuthorizedAction> {
        let Claim {
            tool,
            arguments,
            action,
            receipts,
            reason,
            rollback,
            argument_checks,
            overwrite,
            now,
        } = claim;
        let mut receipts = require_receipts(&tool, receipts)?;
        if reason.trim().is_empty() {
            return Err(CosError::MissingReason { tool });
        }
        let mut rule_checks = vec![RuleCheck {
            rule: StandingRule::ReceiptsPresent,
            evidence: format!("{} receipt(s)", receipts.as_slice().len()),
        }];
        let mut add = |check: RuleCheck, extra: Vec<Receipt>, receipts: &mut Receipts| {
            rule_checks.push(check);
            for receipt in extra {
                if !receipts.as_slice().contains(&receipt) {
                    receipts.push(receipt);
                }
            }
        };
        for (check, receipt) in argument_checks {
            add(check, vec![receipt], &mut receipts);
        }

        // The overwritten file must be the declared target.
        if let Some(path) = &overwrite {
            let names_it = matches!(&action, OperatorAction::Overwrite { target, .. }
                if self.resolve_declared(target) == *path);
            if !names_it {
                return Err(CosError::UnverifiedDeletion {
                    operation: "overwrite",
                    target: path.display().to_string(),
                    gap: DeletionGap::TargetMismatch,
                });
            }
        }

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
                add(check, vec![receipt], &mut receipts);
            }
            OperatorAction::Delete {
                target,
                reversibility,
                observed_digest,
                copy,
            }
            | OperatorAction::Overwrite {
                target,
                reversibility,
                observed_digest,
                copy,
            } => {
                let operation = if matches!(action, OperatorAction::Delete { .. }) {
                    "delete"
                } else {
                    "overwrite"
                };
                let copy = copy.as_deref().map(|c| self.resolve_declared(c));
                let (check, extra) = check_verified_target(
                    self.digests.as_ref(),
                    operation,
                    &self.resolve_declared(target),
                    observed_digest,
                    *reversibility,
                    copy.as_deref(),
                    now,
                )?;
                add(check, extra, &mut receipts);
            }
            OperatorAction::Spend { .. }
            | OperatorAction::Merge { .. }
            | OperatorAction::Tracker { .. }
            | OperatorAction::Outward { .. } => {}
        }

        // Reserve last, once every other check passed, and write the
        // reservation ahead of the call.
        let reservation = match &action {
            OperatorAction::Spend { amount, .. } => {
                let mut ledger = self.ledger.lock().unwrap_or_else(|e| e.into_inner());
                let check = ledger.reserve(amount)?;
                let reservation = SpendReservation {
                    reservation: uuid::Uuid::now_v7().to_string(),
                    tool: tool.clone(),
                    amount: amount.clone(),
                    reserved_at: now,
                };
                if let Err(e) = self
                    .log
                    .append(&CosLogRecord::SpendReserved(reservation.clone()))
                {
                    ledger.release(amount);
                    return Err(e);
                }
                drop(ledger);
                add(check, Vec::new(), &mut receipts);
                Some(reservation)
            }
            _ => None,
        };

        Ok(AuthorizedAction {
            tool,
            arguments: redact_arguments(arguments),
            action,
            receipts,
            reason,
            rollback,
            rule_checks,
            reservation,
        })
    }

    /// Settle an admission once its call ended. For an operator-class action
    /// this settles any spend reservation (returned only when the call
    /// certainly failed) and delivers the announcement.
    pub fn settle(&self, admission: Admission, outcome: ActionOutcome) -> CosResult<Settlement> {
        match admission {
            Admission::Autonomous { tool, tier } => Ok(Settlement::Recorded { tool, tier }),
            Admission::Coordination { tool, receipts } => {
                Ok(Settlement::Coordinated { tool, receipts })
            }
            Admission::OperatorClass(authorized) => {
                let authorized = *authorized;
                let now = (self.clock)();
                let settled = match &authorized.reservation {
                    Some(reservation) => {
                        let spent = !outcome.certainly_failed();
                        if !spent {
                            self.ledger
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .release(&reservation.amount);
                        }
                        self.log
                            .append(&CosLogRecord::SpendSettled(SpendSettlement {
                                reservation: reservation.reservation.clone(),
                                spent,
                                settled_at: now,
                            }))
                    }
                    None => Ok(()),
                };
                let announcement = CosAnnouncement {
                    akh: self.profile.akh.clone(),
                    announced_at: now,
                    tool: authorized.tool,
                    arguments: authorized.arguments,
                    action: authorized.action,
                    receipts: authorized.receipts,
                    reason: authorized.reason,
                    rollback: authorized.rollback,
                    rule_checks: authorized.rule_checks,
                    outcome,
                };
                self.log
                    .append(&CosLogRecord::Announcement(Box::new(announcement.clone())))
                    .map_err(|e| CosError::AnnouncementFailed {
                        sink: self.log.describe(),
                        message: e.to_string(),
                    })?;
                settled?;
                Ok(Settlement::Announced(Box::new(announcement)))
            }
        }
    }
}

/// The inputs of an operator-class authorization.
struct Claim<'a> {
    tool: String,
    arguments: &'a Map<String, Value>,
    action: OperatorAction,
    receipts: Vec<Receipt>,
    reason: String,
    rollback: RollbackPath,
    argument_checks: Vec<(RuleCheck, Receipt)>,
    overwrite: Option<PathBuf>,
    now: UnixSeconds,
}

fn require_receipts(tool: &str, receipts: Vec<Receipt>) -> CosResult<Receipts> {
    Receipts::try_from(receipts).map_err(|_| CosError::MissingReceipts {
        tool: tool.to_string(),
    })
}

/// Parse the `cos_declaration` argument. Models sometimes send nested JSON as
/// a string, so a string holding JSON is accepted too.
fn parse_declaration(tool: &str, raw: Value) -> CosResult<Declaration> {
    let malformed = |message: String| CosError::MalformedDeclaration {
        tool: tool.to_string(),
        message,
    };
    match raw {
        Value::String(text) => serde_json::from_str(&text).map_err(|e| malformed(e.to_string())),
        value => serde_json::from_value(value).map_err(|e| malformed(e.to_string())),
    }
}

/// The `cos_declaration` parameter as advertised in a tool's schema.
fn declaration_param(required: bool) -> ToolParamSchema {
    ToolParamSchema {
        name: DECLARATION_PARAM.into(),
        description: "Chief-of-Staff declaration (skills/chief-of-staff): tier `coordination` \
                      with receipts for T2 posts; tier `operator-class` with action, receipts, \
                      reason and rollback for T3 actions and overwrites of existing files."
            .into(),
        required,
        param_type: "object".into(),
        schema: Some(DECLARATION_SCHEMA.clone()),
    }
}

impl ToolCallGuard for CosGuard {
    fn admit(&self, manifest: &ToolManifest, input: &mut ToolInput) -> Result<GuardTicket, String> {
        let declaration = input
            .params
            .remove(DECLARATION_PARAM)
            .map(|raw| parse_declaration(&manifest.name, raw))
            .transpose()
            .map_err(|e| e.to_tool_refusal())?;
        let admission = CosGuard::admit(self, manifest, declaration, &input.params)
            .map_err(|e| e.to_tool_refusal())?;
        let ticket = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(ticket, admission);
        Ok(GuardTicket(ticket))
    }

    fn settle(
        &self,
        ticket: GuardTicket,
        outcome: CallOutcome,
        detail: &str,
    ) -> Result<(), String> {
        let admission = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&ticket.0)
            .ok_or(CosError::UnknownTicket { ticket: ticket.0 })
            .map_err(|e| e.to_tool_refusal())?;
        let detail = detail.to_string();
        let outcome = match outcome {
            CallOutcome::Succeeded => ActionOutcome::Succeeded { detail },
            CallOutcome::Failed => ActionOutcome::Failed { detail },
            CallOutcome::Aborted => ActionOutcome::Unknown { detail },
        };
        CosGuard::settle(self, admission, outcome)
            .map(|_| ())
            .map_err(|e| e.to_tool_refusal())
    }

    fn advertised_params(&self, manifest: &ToolManifest) -> Vec<ToolParamSchema> {
        match self.profile.classify(manifest).tier() {
            Some(AutonomyTier::Coordination | AutonomyTier::OperatorClass) => {
                vec![declaration_param(true)]
            }
            Some(_) if self.profile.overwrite_spec(&manifest.name).is_some() => {
                vec![declaration_param(false)]
            }
            _ => Vec::new(),
        }
    }
}

// ── Fail-closed fallback ────────────────────────────────────────────────────

/// Installed when the Chief-of-Staff guard cannot load. It admits only tools
/// whose manifests declare themselves read-only. It is a guard, not a
/// permission mode, so switching the mode (e.g. over ACP
/// `_akh/permission/mode`) cannot lift it.
#[derive(Debug)]
pub struct FailClosedGuard {
    cause: String,
}

impl FailClosedGuard {
    /// A fallback remembering why the real guard did not load.
    pub fn new(cause: impl Into<String>) -> Self {
        Self {
            cause: cause.into(),
        }
    }
}

impl ToolCallGuard for FailClosedGuard {
    fn admit(&self, manifest: &ToolManifest, input: &mut ToolInput) -> Result<GuardTicket, String> {
        input.params.remove(DECLARATION_PARAM);
        if manifest.danger.effect.is_read_only() {
            Ok(GuardTicket(0))
        } else {
            Err(CosError::GuardUnavailable {
                tool: manifest.name.clone(),
                cause: self.cause.clone(),
            }
            .to_tool_refusal())
        }
    }

    fn settle(&self, _: GuardTicket, _: CallOutcome, _: &str) -> Result<(), String> {
        Ok(())
    }
}

/// What installing the Chief-of-Staff guard produced.
#[derive(Debug)]
pub enum GuardInstall {
    /// The real guard.
    Active(Arc<CosGuard>),
    /// The guard did not load; the akh runs read-only behind the fallback.
    FailClosed {
        guard: Arc<FailClosedGuard>,
        error: CosError,
    },
}

impl GuardInstall {
    /// Build the guard from `config`, falling back to [`FailClosedGuard`].
    pub fn from_config(config: &ChiefOfStaffConfig, write_roots: &[PathBuf]) -> Self {
        match CosGuard::from_config(config, write_roots) {
            Ok(guard) => Self::Active(Arc::new(guard)),
            Err(error) => Self::FailClosed {
                guard: Arc::new(FailClosedGuard::new(error.to_string())),
                error,
            },
        }
    }

    /// The guard to install on the tool registry.
    pub fn guard(&self) -> Arc<dyn ToolCallGuard> {
        match self {
            Self::Active(guard) => guard.clone(),
            Self::FailClosed { guard, .. } => guard.clone(),
        }
    }
}
