//! Typed `lilith/cos-*@1` contracts (ADR 116 Decision 1).
//!
//! The JSON Schema twins live in `skills/chief-of-staff/schemas/`. A document
//! on the wire is a [`CosDocument`]: the `contract` tag names the schema, the
//! remaining fields are the variant's struct. Invariants the schemas state
//! (at least one receipt, a well-formed SHA, a BCP 47 language tag) are
//! enforced by the newtypes here at deserialization, so an ill-formed document
//! never becomes a value.

use serde::{Deserialize, Serialize};

use super::error::CosError;

// ── Contract envelope ───────────────────────────────────────────────────────

/// One Chief-of-Staff document, tagged by its contract id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "contract")]
pub enum CosDocument {
    /// Done / In flight / Blocked / Next.
    #[serde(rename = "lilith/cos-briefing@1")]
    Briefing(CosBriefing),
    /// Something the operator must do.
    #[serde(rename = "lilith/cos-action-item@1")]
    ActionItem(CosActionItem),
    /// An operator-class action Lilith took, announced after the fact.
    #[serde(rename = "lilith/cos-announcement@1")]
    Announcement(CosAnnouncement),
    /// A commitment Lilith keeps for the operator.
    #[serde(rename = "lilith/cos-commitment@1")]
    Commitment(CosCommitment),
    /// Guidance to an implementer, the PM or the PO.
    #[serde(rename = "lilith/cos-guidance@1")]
    Guidance(CosGuidance),
    /// An action Lilith proposes instead of taking.
    #[serde(rename = "lilith/cos-proposal@1")]
    Proposal(CosProposal),
}

impl CosDocument {
    /// The contract id this document serializes under.
    pub fn contract_id(&self) -> &'static str {
        match self {
            Self::Briefing(_) => "lilith/cos-briefing@1",
            Self::ActionItem(_) => "lilith/cos-action-item@1",
            Self::Announcement(_) => "lilith/cos-announcement@1",
            Self::Commitment(_) => "lilith/cos-commitment@1",
            Self::Guidance(_) => "lilith/cos-guidance@1",
            Self::Proposal(_) => "lilith/cos-proposal@1",
        }
    }
}

// ── Shared value types ──────────────────────────────────────────────────────

/// Seconds since the Unix epoch.
pub type UnixSeconds = u64;

/// A git commit SHA: 7 to 64 lowercase hex digits.
///
/// Parsing rejects anything else, so a value can never smuggle an option or a
/// revision expression into the `git merge-base` call.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CommitSha(String);

impl CommitSha {
    /// The SHA as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for CommitSha {
    type Error = CosError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let well_formed = (7..=64).contains(&value.len())
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if well_formed {
            Ok(Self(value))
        } else {
            Err(CosError::InvalidValue {
                field: "commit sha",
                message: format!("`{value}` is not 7..64 lowercase hex digits"),
            })
        }
    }
}

impl From<CommitSha> for String {
    fn from(value: CommitSha) -> Self {
        value.0
    }
}

impl std::fmt::Display for CommitSha {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A BCP 47 language tag (`de`, `de-CH`, `en`), detected structurally from
/// the operator's own messages.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct LanguageTag(String);

impl LanguageTag {
    /// The tag as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for LanguageTag {
    type Error = CosError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        let mut parts = value.split('-');
        let primary_ok = parts.next().is_some_and(|primary| {
            (2..=3).contains(&primary.len()) && primary.bytes().all(|b| b.is_ascii_lowercase())
        });
        let rest_ok = parts.all(|part| {
            (2..=8).contains(&part.len()) && part.bytes().all(|b| b.is_ascii_alphanumeric())
        });
        if primary_ok && rest_ok {
            Ok(Self(value))
        } else {
            Err(CosError::InvalidValue {
                field: "language",
                message: format!("`{value}` is not a BCP 47 tag such as `de` or `de-CH`"),
            })
        }
    }
}

impl From<LanguageTag> for String {
    fn from(value: LanguageTag) -> Self {
        value.0
    }
}

/// An ISO 4217 currency code (`USD`, `CHF`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CurrencyCode(String);

impl CurrencyCode {
    /// The code as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for CurrencyCode {
    type Error = CosError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 3 && value.bytes().all(|b| b.is_ascii_uppercase()) {
            Ok(Self(value))
        } else {
            Err(CosError::InvalidValue {
                field: "currency",
                message: format!("`{value}` is not a three-letter ISO 4217 code"),
            })
        }
    }
}

impl From<CurrencyCode> for String {
    fn from(value: CurrencyCode) -> Self {
        value.0
    }
}

/// An amount of money in minor units (cents).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Money {
    /// Amount in the currency's minor unit.
    pub minor_units: u64,
    /// The currency.
    pub currency: CurrencyCode,
}

/// What kind of evidence a receipt points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReceiptKind {
    /// A tool call made in this episode (its call id).
    ToolCall,
    /// An Anima work item or plan read in this episode.
    WorkItem,
    /// A pull request.
    PullRequest,
    /// A git commit.
    Commit,
    /// An Agent Board message.
    BoardMessage,
    /// A run record (training run, CI job).
    RunRecord,
    /// A `git merge-base --is-ancestor` verdict.
    AncestryCheck,
    /// A content digest of a file, object or artifact.
    Digest,
    /// A document (ADR, plan, man page, source file).
    Document,
}

/// Evidence read in this episode that backs a claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    /// What the receipt points at.
    pub kind: ReceiptKind,
    /// The stable reference (id, URL, path, SHA).
    pub reference: String,
    /// When the evidence was read.
    pub observed_at: UnixSeconds,
    /// Optional digest of what was read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

/// A non-empty list of receipts (Honesty: no claim without one).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<Receipt>", into = "Vec<Receipt>")]
pub struct Receipts(Vec<Receipt>);

impl Receipts {
    /// The receipts in order.
    pub fn as_slice(&self) -> &[Receipt] {
        &self.0
    }

    /// Append a receipt (the list stays non-empty).
    pub fn push(&mut self, receipt: Receipt) {
        self.0.push(receipt);
    }
}

impl TryFrom<Vec<Receipt>> for Receipts {
    type Error = CosError;

    fn try_from(value: Vec<Receipt>) -> Result<Self, Self::Error> {
        if value.is_empty() {
            Err(CosError::InvalidValue {
                field: "receipts",
                message: "at least one receipt is required".into(),
            })
        } else {
            Ok(Self(value))
        }
    }
}

impl From<Receipts> for Vec<Receipt> {
    fn from(value: Receipts) -> Self {
        value.0
    }
}

/// What kind of entity an item is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EntityKind {
    WorkItem,
    Plan,
    PullRequest,
    Commit,
    Run,
    BoardThread,
    Akh,
    Host,
    Document,
}

/// A reference to a tracked entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntityRef {
    /// The entity's kind.
    pub kind: EntityKind,
    /// Its stable id (WI UUID, PR number, SHA, run id, host name).
    pub id: String,
}

// ── Briefing ────────────────────────────────────────────────────────────────

/// One line of a briefing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BriefingItem {
    /// What the line is about.
    pub entity: EntityRef,
    /// The status in the operator's language.
    pub summary: String,
    /// Where the status was read.
    pub receipts: Receipts,
}

/// `lilith/cos-briefing@1`: Done / In flight / Blocked / Next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosBriefing {
    /// The operator's language.
    pub language: LanguageTag,
    /// When the briefing was produced.
    pub generated_at: UnixSeconds,
    /// Finished since the last briefing.
    pub done: Vec<BriefingItem>,
    /// Moving now.
    pub in_flight: Vec<BriefingItem>,
    /// Parked, failed or waiting on someone (parked items first).
    pub blocked: Vec<BriefingItem>,
    /// What happens next, and who moves it.
    pub next: Vec<BriefingItem>,
}

// ── Action item ─────────────────────────────────────────────────────────────

/// When an action item is due.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Trigger {
    /// Due now.
    Now,
    /// Due by a deadline.
    Deadline {
        /// The deadline.
        due_at: UnixSeconds,
    },
    /// Due when an event happens.
    Event {
        /// The event, in the operator's language.
        description: String,
    },
}

/// `lilith/cos-action-item@1`: something the operator must do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosActionItem {
    /// The operator's language.
    pub language: LanguageTag,
    /// The action.
    pub action: String,
    /// Where to do it (host, file, UI; or that it is not in this repo).
    pub location: String,
    /// When it is due.
    pub trigger: Trigger,
    /// What happens if it is skipped.
    pub cost_of_skipping: String,
    /// Entities involved.
    #[serde(default)]
    pub entities: Vec<EntityRef>,
    /// Evidence that the action is needed.
    pub receipts: Receipts,
}

// ── Operator-class actions ──────────────────────────────────────────────────

/// A tracker mutation Lilith may take (T3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrackerOperation {
    PromoteDraft,
    Unpark,
    CreateWorkItem,
    CancelWorkItem,
    CreatePlan,
    CancelPlan,
}

/// Whether a deletion or overwrite can be undone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Reversibility {
    /// Undo exists (trash, snapshot, versioned bucket, git history).
    Reversible,
    /// No undo without a copy.
    Irreversible,
}

/// An operator-class (T3) action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum OperatorAction {
    /// Deploy an artifact built from `artifact_sha`.
    Deploy {
        artifact_sha: CommitSha,
        target: String,
    },
    /// Restart a service onto an artifact built from `artifact_sha`.
    Restart {
        service: String,
        artifact_sha: CommitSha,
    },
    /// Merge a change (PR) that is green.
    Merge { change: String },
    /// Mutate the tracker (promote, unpark, create or cancel).
    Tracker {
        operation: TrackerOperation,
        target: String,
    },
    /// Spend money (pods, API credit).
    Spend { amount: Money, purpose: String },
    /// Delete something. The runtime computes the target's digest itself and
    /// compares it with `observed_digest` (what she saw when she looked); an
    /// irreversible deletion also needs a `copy` whose runtime-computed digest
    /// matches the target's.
    Delete {
        target: String,
        reversibility: Reversibility,
        observed_digest: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        copy: Option<String>,
    },
    /// Overwrite something (including any file_io/edit write to an existing
    /// file). Verified like [`OperatorAction::Delete`].
    Overwrite {
        target: String,
        reversibility: Reversibility,
        observed_digest: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        copy: Option<String>,
    },
    /// Anything outward-facing outside the fleet.
    Outward { destination: String },
    /// Activate, promote or swap a model adapter. Always outside her grant.
    ActivateAdapter { adapter: String },
}

/// How an action can be rolled back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum RollbackPath {
    /// Undo by concrete steps (revert commit, redeploy previous SHA, re-park).
    Revert { steps: String },
    /// Restore from a verified copy.
    RestoreFromCopy { location: String },
    /// No rollback exists; what limits the damage.
    Irreversible { mitigation: String },
}

/// The standing rules the runtime checks before an operator-class action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StandingRule {
    /// Deploy and restart only artifacts built from an ancestor of main.
    MainOnlyDeploy,
    /// Spend only within the configured budget.
    WithinBudget,
    /// Delete or overwrite only an inspected target, with a verified copy when
    /// irreversible.
    VerifiedDeletion,
    /// Every T2/T3 call carries evidence receipts.
    ReceiptsPresent,
}

impl std::fmt::Display for StandingRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MainOnlyDeploy => "main-only-deploy",
            Self::WithinBudget => "within-budget",
            Self::VerifiedDeletion => "verified-deletion",
            Self::ReceiptsPresent => "receipts-present",
        })
    }
}

/// A standing-rule check that passed, with the evidence it produced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleCheck {
    /// The rule.
    pub rule: StandingRule,
    /// What the check observed (for main-only deploy: the quoted
    /// `git merge-base --is-ancestor` verdict).
    pub evidence: String,
}

/// How the tool call that carried the action ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ActionOutcome {
    /// The tool reported success.
    Succeeded { detail: String },
    /// The tool reported failure.
    Failed { detail: String },
    /// The call did not finish normally (it panicked or unwound), so whether
    /// the action took effect is unknown. A spend stays counted.
    Unknown { detail: String },
}

impl ActionOutcome {
    /// Whether the action took effect.
    pub fn succeeded(&self) -> bool {
        matches!(self, Self::Succeeded { .. })
    }

    /// Whether the action certainly took no effect, so a budget reservation
    /// can be returned.
    pub fn certainly_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }
}

// ── Announcement ────────────────────────────────────────────────────────────

/// `lilith/cos-announcement@1`: an operator-class action, announced with its
/// receipts, reason and rollback path so the operator can see and reverse it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosAnnouncement {
    /// The akh that acted.
    pub akh: String,
    /// When the action settled.
    pub announced_at: UnixSeconds,
    /// The tool that carried the action.
    pub tool: String,
    /// The tool's actual input arguments (declaration stripped, values that
    /// look like credentials redacted), so a mis-declared action is visible.
    pub arguments: serde_json::Map<String, serde_json::Value>,
    /// The action.
    pub action: OperatorAction,
    /// Evidence for the action, including the runtime's own rule receipts.
    pub receipts: Receipts,
    /// Why she acted.
    pub reason: String,
    /// How to undo it.
    pub rollback: RollbackPath,
    /// The standing-rule checks that passed before the action.
    pub rule_checks: Vec<RuleCheck>,
    /// How the action ended.
    pub outcome: ActionOutcome,
}

// ── Commitment ──────────────────────────────────────────────────────────────

/// Who carries a commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CommitmentOwner {
    /// Lilith keeps it herself.
    Lilith,
    /// The operator promised it; Lilith keeps it visible.
    Operator,
    /// Someone else (an akh, implementer, PM, PO); Lilith chases it.
    Delegate,
}

/// Where a commitment stands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "kebab-case", deny_unknown_fields)]
pub enum CommitmentStatus {
    Open,
    AtRisk {
        reason: String,
    },
    Blocked {
        reason: String,
    },
    /// Closed only on evidence.
    Closed {
        evidence: Receipts,
    },
    Dropped {
        reason: String,
    },
}

/// `lilith/cos-commitment@1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosCommitment {
    /// Stable ledger id.
    pub id: String,
    /// The operator's language.
    pub language: LanguageTag,
    /// Who carries it.
    pub owner: CommitmentOwner,
    /// The counterparty when owned by a delegate (akh name, board agent id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterparty: Option<String>,
    /// What was committed.
    pub statement: String,
    /// When it is due.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub due_at: Option<UnixSeconds>,
    /// The evidence that will close it.
    pub closing_evidence: String,
    /// Current status.
    pub status: CommitmentStatus,
    /// Entities involved.
    #[serde(default)]
    pub entities: Vec<EntityRef>,
    /// Where the commitment was made.
    pub receipts: Receipts,
}

// ── Guidance ────────────────────────────────────────────────────────────────

/// Who the guidance is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GuidanceAudience {
    Implementer,
    ProductManager,
    ProductOwner,
}

/// The illumos subsystem a finding concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IllumosSubsystem {
    Smf,
    Ips,
    Zones,
    Zfs,
    Rbac,
    LibcAbi,
    Dtrace,
    /// A Linux or Solaris assumption trap.
    PortabilityTrap,
    /// Scope, acceptance criteria or dependency order (PM/PO steering).
    Scope,
    Other,
}

/// How much the finding matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GuidanceSeverity {
    Note,
    Concern,
    Blocker,
}

/// What a citation points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CitationKind {
    /// A man page (`smf(7)`, `zfs(8)`).
    ManPage,
    /// A source file and line.
    Source,
    /// A native check in the OpenIndiana validation VM.
    VmCheck,
    /// An ADR, plan or other document.
    Document,
}

/// A grounding citation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Citation {
    /// What it points at.
    pub kind: CitationKind,
    /// The reference (`zfs-allow(8)`, `usr/src/lib/libc/port/gen/foo.c:42`).
    pub reference: String,
}

/// `lilith/cos-guidance@1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosGuidance {
    /// The recipient's language.
    pub language: LanguageTag,
    /// Who it is for.
    pub audience: GuidanceAudience,
    /// The WI, plan or PR it is about.
    pub subject: EntityRef,
    /// The subsystem.
    pub subsystem: IllumosSubsystem,
    /// How much it matters.
    pub severity: GuidanceSeverity,
    /// What is wrong or missing.
    pub finding: String,
    /// What to do instead.
    pub recommendation: String,
    /// Grounding (at least one).
    pub citations: Vec<Citation>,
    /// Evidence read in this episode.
    pub receipts: Receipts,
}

// ── Proposal ────────────────────────────────────────────────────────────────

/// Why Lilith proposes instead of acting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProposalMotive {
    /// Her own choice: she is unsure.
    Unsure,
    /// The operator put the tier behind a gate.
    TierGated,
    /// The action is outside her grant (adapter activation).
    OutsideGrant,
}

/// `lilith/cos-proposal@1`: an action proposed instead of taken.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosProposal {
    /// The operator's language.
    pub language: LanguageTag,
    /// Why she proposes.
    pub motive: ProposalMotive,
    /// The action.
    pub action: OperatorAction,
    /// Evidence.
    pub receipts: Receipts,
    /// Why the action is warranted.
    pub reason: String,
    /// How it would be undone.
    pub rollback: RollbackPath,
}
