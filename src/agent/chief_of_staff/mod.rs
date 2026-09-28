//! Lilith as Chief of Staff: role contracts, permission profile and standing
//! rules.
//!
//! - ADR 116 (Decisions 1, 2, 5; revision 4 = full autonomy):
//!   `docs/ai/decisions/116-lilith-chief-of-staff-guided-rl.md`
//! - Plan slice C1: `docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md`,
//!   Anima WI `01a0e270-0288-7ac1-b297-a45375d4e6e6`
//! - Skill pack (procedures, profile, JSON Schemas): `skills/chief-of-staff/`
//!
//! Every tier (T0 read … T3 operator-class) is autonomous. The runtime does
//! not ask for approval; it enforces the standing rules (main-only deploy,
//! budget, verified deletion), requires receipts on every T2/T3 call, refuses
//! adapter activation (outside her grant), and announces each operator-class
//! action with its receipts, reason and rollback path. The operator can put
//! any tier back behind a gate in the profile.
//!
//! [`CosGuard`] plugs into the tool registry as a
//! [`ToolCallGuard`](crate::agent::permission::ToolCallGuard); an akh opts in
//! with a `[chief_of_staff]` table ([`ChiefOfStaffConfig`]).

pub mod contracts;
pub mod error;
pub mod guard;
pub mod policy;

#[cfg(test)]
mod tests;

pub use contracts::{
    ActionOutcome, CommitSha, CosActionItem, CosAnnouncement, CosBriefing, CosCommitment,
    CosDocument, CosGuidance, CosProposal, OperatorAction, Receipt, Receipts, RollbackPath,
    RuleCheck, StandingRule,
};
pub use error::{CosError, CosResult, DeletionGap};
pub use guard::{
    Admission, AnnouncementSink, AuthorizedAction, CosGuard, JsonlAnnouncementSink,
    MemoryAnnouncementSink, Settlement,
};
pub use policy::{
    AncestryOracle, AutonomyTier, ChiefOfStaffConfig, CosPermissionProfile, DECLARATION_PARAM,
    Declaration, GitAncestry, SpendLedger, TierGrant, ToolClass,
};
