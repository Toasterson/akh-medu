//! Lilith's permission profile and standing-rule checks (ADR 116 Decision 1,
//! revision 4).
//!
//! Every tier is autonomous. What binds her is data: the profile classifies
//! each tool into a tier (or outside her grant), the operator can put any tier
//! back behind a gate, and three standing rules are checked at runtime before
//! an operator-class action runs:
//!
//! - **main-only deploy**: a deploy or restart names an artifact SHA that
//!   `git merge-base --is-ancestor` confirms is on the main ref;
//! - **within budget**: a spend fits the remaining configured budget;
//! - **verified deletion**: a deletion or overwrite names an inspected
//!   target, and an irreversible one a verified copy with the same digest.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::agent::tool_manifest::ToolManifest;

use super::contracts::{
    CommitSha, CopyVerification, CosAnnouncement, Money, OperatorAction, Receipt, ReceiptKind,
    Reversibility, RollbackPath, RuleCheck, StandingRule, TargetVerification, UnixSeconds,
};
use super::error::{CosError, CosResult, DeletionGap};

/// The profile contract id.
pub const PROFILE_CONTRACT: &str = "lilith/cos-permission-profile@1";

/// The tool argument that carries a T2/T3 declaration. The guard strips it
/// before the tool runs.
pub const DECLARATION_PARAM: &str = "cos_declaration";

// ── Tiers, grants, tool classes ─────────────────────────────────────────────

/// ADR 116 autonomy tiers, ordered by consequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AutonomyTier {
    /// Read everything in scope.
    #[serde(rename = "t0-read")]
    Read,
    /// Her own ledger, notes, memory and briefings.
    #[serde(rename = "t1-own-state")]
    OwnState,
    /// Chase and guide others; every post carries receipts.
    #[serde(rename = "t2-coordination")]
    Coordination,
    /// Operator-class actions; each is announced after the fact.
    #[serde(rename = "t3-operator-class")]
    OperatorClass,
}

impl AutonomyTier {
    /// All tiers in order.
    pub const ALL: [Self; 4] = [
        Self::Read,
        Self::OwnState,
        Self::Coordination,
        Self::OperatorClass,
    ];
}

impl std::fmt::Display for AutonomyTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Read => "T0 read",
            Self::OwnState => "T1 own state",
            Self::Coordination => "T2 coordination",
            Self::OperatorClass => "T3 operator-class",
        })
    }
}

/// How a tier is granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TierGrant {
    /// She acts without asking (ADR 116 revision 4 default for every tier).
    Autonomous,
    /// The operator took the tier back: she proposes instead of acting.
    Gated,
}

/// How the profile classifies one tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolClass {
    /// T0.
    T0Read,
    /// T1.
    T1OwnState,
    /// T2.
    T2Coordination,
    /// T3.
    T3OperatorClass,
    /// Never hers to call (adapter activation).
    OutsideGrant,
}

impl ToolClass {
    /// The tier, or `None` when the tool is outside her grant.
    pub fn tier(self) -> Option<AutonomyTier> {
        match self {
            Self::T0Read => Some(AutonomyTier::Read),
            Self::T1OwnState => Some(AutonomyTier::OwnState),
            Self::T2Coordination => Some(AutonomyTier::Coordination),
            Self::T3OperatorClass => Some(AutonomyTier::OperatorClass),
            Self::OutsideGrant => None,
        }
    }
}

/// `lilith/cos-permission-profile@1`: the data that binds Lilith's tools to
/// tiers. Tool keys are exact names or `prefix*` patterns; the exact name, then
/// the longest matching prefix, wins. An unclassified tool is T0 when its
/// manifest declares itself read-only and T3 otherwise (fail closed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosPermissionProfile {
    /// Must be [`PROFILE_CONTRACT`].
    pub contract: String,
    /// The akh the profile is for.
    pub akh: String,
    /// The ref deploys must descend from (`origin/main`).
    pub main_ref: String,
    /// Grant per tier. A missing tier is gated (fail closed).
    pub grants: BTreeMap<AutonomyTier, TierGrant>,
    /// Tool name or `prefix*` pattern to class.
    pub tools: BTreeMap<String, ToolClass>,
}

impl CosPermissionProfile {
    /// Parse and validate a profile from JSON text.
    pub fn from_json(path: &str, text: &str) -> CosResult<Self> {
        let profile: Self = serde_json::from_str(text).map_err(|e| CosError::ProfileParse {
            path: path.to_string(),
            message: e.to_string(),
        })?;
        profile.validate(path)?;
        Ok(profile)
    }

    /// Read a profile file.
    pub fn load(path: &Path) -> CosResult<Self> {
        let display = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| CosError::ProfileRead {
            path: display.clone(),
            source,
        })?;
        Self::from_json(&display, &text)
    }

    fn validate(&self, path: &str) -> CosResult<()> {
        let invalid = |message: String| CosError::ProfileParse {
            path: path.to_string(),
            message,
        };
        if self.contract != PROFILE_CONTRACT {
            return Err(invalid(format!(
                "contract is `{}`, expected `{PROFILE_CONTRACT}`",
                self.contract
            )));
        }
        if self.main_ref.is_empty() || self.main_ref.starts_with('-') {
            return Err(invalid(format!(
                "main_ref `{}` must be a ref name such as `origin/main`",
                self.main_ref
            )));
        }
        if let Some(bad) = self
            .tools
            .keys()
            .find(|key| key.is_empty() || key.trim_end_matches('*').contains('*'))
        {
            return Err(invalid(format!(
                "tool key `{bad}` must be a name or a single trailing `prefix*` pattern"
            )));
        }
        Ok(())
    }

    /// The grant for `tier` (missing ⇒ gated).
    pub fn grant(&self, tier: AutonomyTier) -> TierGrant {
        self.grants.get(&tier).copied().unwrap_or(TierGrant::Gated)
    }

    /// Classify a tool by its manifest.
    pub fn classify(&self, manifest: &ToolManifest) -> ToolClass {
        if let Some(class) = self.tools.get(&manifest.name) {
            return *class;
        }
        let by_prefix = self
            .tools
            .iter()
            .filter_map(|(key, class)| {
                key.strip_suffix('*')
                    .filter(|prefix| manifest.name.starts_with(prefix))
                    .map(|prefix| (prefix.len(), *class))
            })
            .max_by_key(|(len, _)| *len);
        match by_prefix {
            Some((_, class)) => class,
            None if manifest.danger.effect.is_read_only() => ToolClass::T0Read,
            None => ToolClass::T3OperatorClass,
        }
    }
}

// ── Per-akh configuration (`[chief_of_staff]` in akh.toml) ──────────────────

/// The `[chief_of_staff]` table of an akh's `akh.toml`.
///
/// ```toml
/// [chief_of_staff]
/// profile = "/srv/akh-medu/skills/chief-of-staff/permission-profile.json"
/// repository = "/srv/akh-medu"
/// announcements = "/var/lib/akh/lilith/cos-announcements.jsonl"
/// budget = { minor_units = 6673, currency = "USD" }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChiefOfStaffConfig {
    /// The permission profile JSON.
    pub profile: PathBuf,
    /// The git checkout the ancestry check runs in.
    pub repository: PathBuf,
    /// The announcement sink: a JSONL file the CoS-thread relay reads.
    pub announcements: PathBuf,
    /// The spend ceiling she is given.
    pub budget: Money,
}

// ── Declarations ────────────────────────────────────────────────────────────

/// What Lilith attaches to a T2/T3 tool call (`cos_declaration`).
///
/// A short-lived parse value (consumed by admission), so the size difference
/// between the variants does not matter.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "tier", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Declaration {
    /// A T2 post.
    Coordination {
        /// Evidence the post rests on.
        #[serde(default)]
        receipts: Vec<Receipt>,
    },
    /// A T3 action.
    OperatorClass {
        /// The action.
        action: OperatorAction,
        /// Evidence the action rests on.
        #[serde(default)]
        receipts: Vec<Receipt>,
        /// Why now.
        reason: String,
        /// How to undo it.
        rollback: RollbackPath,
    },
}

impl Declaration {
    /// The tier the declaration claims.
    pub fn tier(&self) -> AutonomyTier {
        match self {
            Self::Coordination { .. } => AutonomyTier::Coordination,
            Self::OperatorClass { .. } => AutonomyTier::OperatorClass,
        }
    }
}

// ── Standing rule: main-only deploy ─────────────────────────────────────────

/// Decides whether a commit is an ancestor of the main ref.
pub trait AncestryOracle: Send + Sync {
    /// `Ok(true)` if `sha` is an ancestor of `main_ref`, `Ok(false)` if not.
    fn is_ancestor(&self, sha: &CommitSha, main_ref: &str) -> CosResult<bool>;
}

/// [`AncestryOracle`] backed by `git merge-base --is-ancestor` in a checkout.
///
/// It reads the local main ref and does not fetch: a stale ref can only refuse
/// a deploy that would have been allowed, never allow one from a branch.
#[derive(Debug, Clone)]
pub struct GitAncestry {
    repository: PathBuf,
}

impl GitAncestry {
    /// Check ancestry in `repository`.
    pub fn new(repository: impl Into<PathBuf>) -> Self {
        Self {
            repository: repository.into(),
        }
    }
}

impl AncestryOracle for GitAncestry {
    fn is_ancestor(&self, sha: &CommitSha, main_ref: &str) -> CosResult<bool> {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(&self.repository)
            .args(["merge-base", "--is-ancestor", sha.as_str(), main_ref])
            .output()
            .map_err(|source| CosError::AncestryIo {
                repository: self.repository.display().to_string(),
                source,
            })?;
        match output.status.code() {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => Err(CosError::AncestryUnknown {
                sha: sha.to_string(),
                main_ref: main_ref.to_string(),
                detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            }),
        }
    }
}

/// Check the main-only-deploy rule. On success, returns the rule check and
/// the receipt quoting the ancestry verdict.
pub fn check_main_only(
    oracle: &dyn AncestryOracle,
    sha: &CommitSha,
    main_ref: &str,
    now: UnixSeconds,
) -> CosResult<(RuleCheck, Receipt)> {
    let command = format!("git merge-base --is-ancestor {sha} {main_ref}");
    if oracle.is_ancestor(sha, main_ref)? {
        Ok((
            RuleCheck {
                rule: StandingRule::MainOnlyDeploy,
                evidence: format!("{command}: exit 0 (ancestor)"),
            },
            Receipt {
                kind: ReceiptKind::AncestryCheck,
                reference: command,
                observed_at: now,
                digest: None,
            },
        ))
    } else {
        Err(CosError::NonMainArtifact {
            sha: sha.to_string(),
            main_ref: main_ref.to_string(),
        })
    }
}

// ── Standing rule: within budget ────────────────────────────────────────────

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

    /// Rebuild the ledger from past announcements (successful spends in the
    /// budget's currency), so a restart does not reset the budget.
    pub fn replay<'a>(
        limit: Money,
        announcements: impl IntoIterator<Item = &'a CosAnnouncement>,
    ) -> Self {
        let spent = announcements
            .into_iter()
            .filter(|a| a.outcome.succeeded())
            .filter_map(|a| match &a.action {
                OperatorAction::Spend { amount, .. } if amount.currency == limit.currency => {
                    Some(amount.minor_units)
                }
                _ => None,
            })
            .fold(0u64, u64::saturating_add);
        Self {
            limit,
            spent_minor_units: spent,
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

    /// Return a reservation whose action did not take effect.
    pub fn release(&mut self, amount: &Money) {
        if amount.currency == self.limit.currency {
            self.spent_minor_units = self.spent_minor_units.saturating_sub(amount.minor_units);
        }
    }
}

// ── Standing rule: verified deletion ────────────────────────────────────────

/// Check the verified-deletion rule for a delete or overwrite.
pub fn check_verified_deletion(
    operation: &'static str,
    target: &TargetVerification,
    reversibility: Reversibility,
    copy: Option<&CopyVerification>,
) -> CosResult<RuleCheck> {
    let refuse = |gap| CosError::UnverifiedDeletion {
        operation,
        target: target.target.clone(),
        gap,
    };
    if target.digest.trim().is_empty() {
        return Err(refuse(DeletionGap::TargetNotInspected));
    }
    let evidence = match (reversibility, copy) {
        (Reversibility::Irreversible, None) => return Err(refuse(DeletionGap::NoVerifiedCopy)),
        (_, Some(copy)) if copy.location == target.target => {
            return Err(refuse(DeletionGap::CopyIsTarget));
        }
        (_, Some(copy)) if copy.digest != target.digest => {
            return Err(refuse(DeletionGap::CopyDigestMismatch));
        }
        (_, Some(copy)) => format!(
            "target {} inspected (digest {}); copy verified at {}",
            target.target, target.digest, copy.location
        ),
        (Reversibility::Reversible, None) => format!(
            "target {} inspected (digest {}); reversible",
            target.target, target.digest
        ),
    };
    Ok(RuleCheck {
        rule: StandingRule::VerifiedDeletion,
        evidence,
    })
}
