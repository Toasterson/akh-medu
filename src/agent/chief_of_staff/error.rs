//! Diagnostics for Lilith's Chief-of-Staff permission profile and standing
//! rules (ADR 116). Every refusal names the options the caller has, so the
//! refusal is something she can act on rather than a dead end.

use miette::Diagnostic;
use thiserror::Error;

use super::contracts::StandingRule;
use super::policy::AutonomyTier;

/// Result alias for the Chief-of-Staff subsystem.
pub type CosResult<T> = Result<T, CosError>;

/// Why a deletion or overwrite was refused under the verified-deletion rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeletionGap {
    /// The target was never inspected (no digest recorded).
    TargetNotInspected,
    /// The action is irreversible and no copy was verified.
    NoVerifiedCopy,
    /// The copy's digest differs from the inspected target's digest.
    CopyDigestMismatch,
    /// The "copy" is the target itself, so deleting the target deletes it too.
    CopyIsTarget,
}

impl std::fmt::Display for DeletionGap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TargetNotInspected => "the target was not inspected",
            Self::NoVerifiedCopy => "the action is irreversible and no copy was verified",
            Self::CopyDigestMismatch => "the copy's digest differs from the target's digest",
            Self::CopyIsTarget => "the copy location is the target itself",
        })
    }
}

/// Errors and refusals of the Chief-of-Staff policy layer.
#[derive(Debug, Error, Diagnostic)]
pub enum CosError {
    #[error("cannot read the Chief-of-Staff profile {path}: {source}")]
    #[diagnostic(
        code(akh::cos::profile_read),
        help(
            "Options: fix the `profile` path in the akh's `[chief_of_staff]` table, or restore \
             `skills/chief-of-staff/permission-profile.json`. Until the profile loads the akh \
             runs read-only (plan mode)."
        )
    )]
    ProfileRead {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid Chief-of-Staff profile {path}: {message}")]
    #[diagnostic(
        code(akh::cos::profile_parse),
        help(
            "Options: correct the JSON against `lilith/cos-permission-profile@1` (tiers are \
             `t0-read`..`t3-operator-class`, grants `autonomous`/`gated`, tool classes add \
             `outside-grant`), or revert the profile to the version on origin/main."
        )
    )]
    ProfileParse { path: String, message: String },

    #[error("invalid {field}: {message}")]
    #[diagnostic(
        code(akh::cos::invalid_value),
        help("Options: correct the value to the documented form and resend the call.")
    )]
    InvalidValue {
        field: &'static str,
        message: String,
    },

    #[error("tool \"{tool}\" is {tier}, and the operator has put that tier behind a gate")]
    #[diagnostic(
        code(akh::cos::tier_gated),
        help(
            "Options: post a `lilith/cos-proposal@1` on the operator's CoS thread describing the \
             action, receipts, reason and rollback path; or ask the operator to restore the \
             autonomous grant for this tier in the permission profile."
        )
    )]
    TierGated { tool: String, tier: AutonomyTier },

    #[error("\"{what}\" is outside Lilith's grant (tool \"{tool}\")")]
    #[diagnostic(
        code(akh::cos::outside_grant),
        help(
            "Adapter activation stays with the operator. Options: post a `lilith/cos-proposal@1` \
             with the adapter, its evaluation receipts and the rollback path, or add it to your \
             next briefing's Next bucket as an operator action item."
        )
    )]
    OutsideGrant { tool: String, what: String },

    #[error("tool \"{tool}\" is {tier} and was called without a `cos_declaration`")]
    #[diagnostic(
        code(akh::cos::missing_declaration),
        help(
            "Options: resend the call with a `cos_declaration` argument. Use \
             {{\"tier\":\"coordination\",\"receipts\":[..]}} for T2, and \
             {{\"tier\":\"operator-class\",\"action\":{{..}},\"receipts\":[..],\"reason\":\"..\",\
             \"rollback\":{{..}}}} for T3. Or use a read-only tool if you only need to look."
        )
    )]
    MissingDeclaration { tool: String, tier: AutonomyTier },

    #[error("malformed `cos_declaration` on tool \"{tool}\": {message}")]
    #[diagnostic(
        code(akh::cos::malformed_declaration),
        help(
            "Options: fix the declaration against the chief-of-staff skill (`tier`, and for \
             operator-class also `action`, `receipts`, `reason`, `rollback`), then resend."
        )
    )]
    MalformedDeclaration { tool: String, message: String },

    #[error("tool \"{tool}\" is {class} but was declared as {declared}")]
    #[diagnostic(
        code(akh::cos::declaration_below_tier),
        help(
            "Options: redeclare the call at the tool's own tier or higher. An operator-class \
             tool needs an operator-class declaration with action, reason and rollback path."
        )
    )]
    DeclarationBelowTier {
        tool: String,
        class: AutonomyTier,
        declared: AutonomyTier,
    },

    #[error("tool \"{tool}\" carries no evidence receipts")]
    #[diagnostic(
        code(akh::cos::missing_receipts),
        help(
            "Honesty: no claim without a receipt. Options: read the evidence first (a T0 tool \
             call in this episode) and cite it in `receipts`, or leave the claim out of the post."
        )
    )]
    MissingReceipts { tool: String },

    #[error("tool \"{tool}\" is operator-class and gives no reason")]
    #[diagnostic(
        code(akh::cos::missing_reason),
        help("Options: state why the action is warranted now, then resend the call.")
    )]
    MissingReason { tool: String },

    #[error("refusing to deploy or restart {sha}: it is not an ancestor of {main_ref}")]
    #[diagnostic(
        code(akh::cos::non_main_artifact),
        help(
            "Deploy only from origin/main. Options: (1) merge the change to main first (open or \
             merge its PR), then build and deploy that main SHA; (2) if the commit did land, \
             `git fetch origin` in the checkout and retry, since the check reads the local \
             {main_ref}; (3) deploy the current main SHA instead. Never hand-build from a branch."
        )
    )]
    NonMainArtifact { sha: String, main_ref: String },

    #[error("ancestry of {sha} against {main_ref} could not be decided: {detail}")]
    #[diagnostic(
        code(akh::cos::ancestry_unknown),
        help(
            "Options: `git fetch origin` in the configured repository and retry; check that the \
             SHA exists (a typo or a commit from another repository); or point \
             `[chief_of_staff] repository` at the right checkout."
        )
    )]
    AncestryUnknown {
        sha: String,
        main_ref: String,
        detail: String,
    },

    #[error("cannot run the ancestry check in {repository}: {source}")]
    #[diagnostic(
        code(akh::cos::ancestry_io),
        help(
            "Options: install git on the host, or fix `[chief_of_staff] repository`. The deploy \
             stays refused until the check can run."
        )
    )]
    AncestryIo {
        repository: String,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "spend of {requested} {currency} exceeds the remaining budget ({remaining} of {limit} \
         {currency}, minor units)"
    )]
    #[diagnostic(
        code(akh::cos::budget_exceeded),
        help(
            "Spend only within the given budget. Options: reduce the spend (a smaller or shorter \
             resource), stop a running resource first, or put a budget increase in front of the \
             operator as a `lilith/cos-action-item@1`. You never add funds."
        )
    )]
    BudgetExceeded {
        requested: u64,
        remaining: u64,
        limit: u64,
        currency: String,
    },

    #[error("spend is in {requested}, the budget is in {budget}")]
    #[diagnostic(
        code(akh::cos::currency_mismatch),
        help("Options: restate the spend in the budget's currency, then resend.")
    )]
    CurrencyMismatch { requested: String, budget: String },

    #[error("refusing to {operation} {target}: {gap}")]
    #[diagnostic(
        code(akh::cos::unverified_deletion),
        help(
            "Look at the target before deleting or overwriting it. Options: inspect it and \
             record its digest in `target`; for anything irreversible, copy it elsewhere and \
             verify the copy's digest in `copy`; or pick a reversible path (move or rename) \
             and declare `reversible`."
        )
    )]
    UnverifiedDeletion {
        operation: &'static str,
        target: String,
        gap: DeletionGap,
    },

    #[error("announcement to {sink} failed: {message}")]
    #[diagnostic(
        code(akh::cos::announcement_failed),
        help(
            "The action already ran. Options: repost the announcement by hand from the audit \
             log, fix the sink (path or permissions), and mention the gap in the next briefing."
        )
    )]
    AnnouncementFailed { sink: String, message: String },

    #[error("announcement log {path} is unreadable at line {line}: {message}")]
    #[diagnostic(
        code(akh::cos::announcement_log),
        help(
            "The spend ledger is rebuilt from this log, so the akh will not start its \
             Chief-of-Staff guard on a damaged one. Options: repair the line (each line is one \
             `lilith/cos-announcement@1` document), or move the log aside after recording the \
             spend it held and restart with a reduced budget."
        )
    )]
    AnnouncementLog {
        path: String,
        line: usize,
        message: String,
    },

    #[error("unknown guard ticket {ticket}")]
    #[diagnostic(
        code(akh::cos::unknown_ticket),
        help("A settle call must use the ticket its admit call returned, exactly once.")
    )]
    UnknownTicket { ticket: u64 },
}

impl CosError {
    /// The standing rule a refusal enforces, if it is a standing-rule refusal.
    pub fn standing_rule(&self) -> Option<StandingRule> {
        match self {
            Self::NonMainArtifact { .. }
            | Self::AncestryUnknown { .. }
            | Self::AncestryIo { .. } => Some(StandingRule::MainOnlyDeploy),
            Self::BudgetExceeded { .. } | Self::CurrencyMismatch { .. } => {
                Some(StandingRule::WithinBudget)
            }
            Self::UnverifiedDeletion { .. } => Some(StandingRule::VerifiedDeletion),
            Self::MissingReceipts { .. } => Some(StandingRule::ReceiptsPresent),
            _ => None,
        }
    }

    /// Render the diagnostic as one line for a refused tool call: the message,
    /// its code and the caller's options.
    pub fn to_tool_refusal(&self) -> String {
        let code = self
            .code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "akh::cos".to_string());
        match self.help() {
            Some(help) => format!("{self} [{code}] {help}"),
            None => format!("{self} [{code}]"),
        }
    }
}
