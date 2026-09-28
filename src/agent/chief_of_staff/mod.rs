//! Lilith as Chief of Staff: role contracts, permission profile and standing
//! rules.
//!
//! - ADR 116 (Decisions 1, 2, 5; revision 4 = full autonomy):
//!   `docs/ai/decisions/116-lilith-chief-of-staff-guided-rl.md`
//! - Plan slice C1: `docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md`,
//!   Anima WI `01a0e270-0288-7ac1-b297-a45375d4e6e6`
//! - Skill pack (procedures, profile, JSON Schemas): `skills/chief-of-staff/`
//!
//! Every tier (T0 read to T3 operator-class) is autonomous. The runtime does
//! not ask for approval. It does five things:
//!
//! - enforces the standing rules (main-only deploy, budget, verified target),
//!   checked against the call's real arguments and not only the declared
//!   action;
//! - requires receipts on every T2/T3 call;
//! - refuses adapter activation, which is outside her grant;
//! - writes spend reservations ahead of the call;
//! - announces each operator-class action with its arguments, receipts,
//!   reason and rollback path.
//!
//! The operator can put any tier back behind a gate in the profile.
//!
//! [`CosGuard`] plugs into the tool registry as a
//! [`ToolCallGuard`](crate::agent::permission::ToolCallGuard). An akh opts in
//! with a `[chief_of_staff]` table ([`ChiefOfStaffConfig`]). If the guard
//! cannot load, [`GuardInstall`] installs the read-only [`FailClosedGuard`]
//! instead, and no permission mode can lift that.

pub mod contracts;
pub mod error;
pub mod guard;
pub mod ledger;
pub mod policy;
pub mod redact;

#[cfg(test)]
mod tests;

pub use contracts::{
    ActionOutcome, CommitSha, CosActionItem, CosAnnouncement, CosBriefing, CosCommitment,
    CosDocument, CosGuidance, CosProposal, OperatorAction, Receipt, Receipts, RollbackPath,
    RuleCheck, StandingRule,
};
pub use error::{CosError, CosResult, DeletionGap, RefSource};
pub use guard::{Admission, AuthorizedAction, CosGuard, FailClosedGuard, GuardInstall, Settlement};
pub use ledger::{CosLog, CosLogRecord, JsonlLog, MemoryLog, SpendLedger};
pub use policy::{
    Ancestry, AncestryOracle, AutonomyTier, ChiefOfStaffConfig, CosPermissionProfile,
    DECLARATION_PARAM, Declaration, DigestOracle, GitAncestry, LocalDigest, TierGrant, ToolClass,
};
