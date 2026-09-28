//! Lilith's permission profile and standing-rule checks (ADR 116 Decision 1,
//! revision 4).
//!
//! Every tier is autonomous. What binds her is data. The profile classifies
//! each tool into a tier (or outside her grant), and the operator can put any
//! tier back behind a gate. The rules stay declarative (she declares what she
//! is doing), but the runtime checks the declaration against what the call
//! actually carries:
//!
//! - **main-only deploy**: a declared deploy/restart SHA, *and* every commit
//!   referenced in the arguments of a deploy-capable tool, must pass
//!   `git merge-base --is-ancestor` against the main ref;
//! - **within budget**: a spend fits the remaining budget (see
//!   [`super::ledger`] for the write-ahead ledger);
//! - **verified target**: before a deletion or overwrite (including a
//!   file_io/edit write to an existing file) the runtime digests the target
//!   itself, compares it with the digest she observed, and for anything
//!   irreversible digests her named copy too;
//! - **adapter activation** is outside her grant, whether by tool name,
//!   declared action, or an adapter marker (from the profile) in the
//!   arguments.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::agent::tool_manifest::ToolManifest;

use super::contracts::{
    CommitSha, Money, OperatorAction, Receipt, ReceiptKind, Reversibility, RollbackPath, RuleCheck,
    StandingRule, UnixSeconds,
};
use super::error::{CosError, CosResult, DeletionGap, RefSource};

/// The profile contract id.
pub const PROFILE_CONTRACT: &str = "lilith/cos-permission-profile@1";

/// The tool argument that carries a T2/T3 declaration. The guard strips it
/// before the tool runs.
pub const DECLARATION_PARAM: &str = "cos_declaration";

/// Most commit-like references checked in one call's arguments. More than
/// this is refused rather than partially checked.
const MAX_ARGUMENT_REFS: usize = 32;

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

/// A condition on one string argument (`action == "write"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParamEquals {
    /// The argument name.
    pub param: String,
    /// The value that makes the condition true.
    pub equals: String,
}

/// How to recognise a file write that can overwrite an existing file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverwriteSpec {
    /// The argument holding the file path.
    pub path_param: String,
    /// Only calls where this holds are writes (absent: every call writes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<ParamEquals>,
}

impl OverwriteSpec {
    /// The path this call writes, if it is a write.
    pub fn written_path<'a>(&self, arguments: &'a Map<String, Value>) -> Option<&'a str> {
        let writes = self.when.as_ref().is_none_or(|when| {
            arguments.get(&when.param).and_then(Value::as_str) == Some(when.equals.as_str())
        });
        writes
            .then(|| arguments.get(&self.path_param).and_then(Value::as_str))
            .flatten()
    }
}

/// Whether `pattern` (exact name or single trailing `prefix*`) matches
/// `name`, and how specific the match is (exact beats any prefix).
fn pattern_match(pattern: &str, name: &str) -> Option<usize> {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix).then_some(prefix.len()),
        None => (pattern == name).then_some(usize::MAX),
    }
}

fn best_match<'a, T>(entries: impl IntoIterator<Item = (&'a String, T)>, name: &str) -> Option<T> {
    entries
        .into_iter()
        .filter_map(|(pattern, value)| pattern_match(pattern, name).map(|rank| (rank, value)))
        .max_by_key(|(rank, _)| *rank)
        .map(|(_, value)| value)
}

/// `lilith/cos-permission-profile@1`: the data that binds Lilith's tools to
/// tiers and the standing rules. Tool keys are exact names or `prefix*`
/// patterns; the exact name, then the longest matching prefix, wins. An
/// unclassified tool is T0 when its manifest declares itself read-only and
/// T3 otherwise (fail closed).
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
    /// Deploy-capable tools (names or `prefix*`): every commit referenced in
    /// their arguments must be on the main ref, whatever the declared action.
    #[serde(default)]
    pub artifact_tools: Vec<String>,
    /// Structural identifiers of adapter activation (file names, endpoints,
    /// flags). Any of them in a T2/T3 call's arguments puts the call outside
    /// her grant.
    #[serde(default)]
    pub adapter_markers: Vec<String>,
    /// File-writing tools whose write to an existing file is an overwrite.
    #[serde(default)]
    pub overwrite_tools: BTreeMap<String, OverwriteSpec>,
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
        let patterns = self
            .tools
            .keys()
            .chain(self.overwrite_tools.keys())
            .chain(self.artifact_tools.iter());
        for key in patterns {
            if key.is_empty() || key.trim_end_matches('*').contains('*') {
                return Err(invalid(format!(
                    "tool key `{key}` must be a name or a single trailing `prefix*` pattern"
                )));
            }
        }
        if let Some(empty) = self.adapter_markers.iter().find(|m| m.trim().is_empty()) {
            return Err(invalid(format!("adapter marker `{empty}` is empty")));
        }
        Ok(())
    }

    /// The grant for `tier` (missing ⇒ gated).
    pub fn grant(&self, tier: AutonomyTier) -> TierGrant {
        self.grants.get(&tier).copied().unwrap_or(TierGrant::Gated)
    }

    /// Classify a tool by its manifest.
    pub fn classify(&self, manifest: &ToolManifest) -> ToolClass {
        match best_match(self.tools.iter().map(|(k, v)| (k, *v)), &manifest.name) {
            Some(class) => class,
            None if manifest.danger.effect.is_read_only() => ToolClass::T0Read,
            None => ToolClass::T3OperatorClass,
        }
    }

    /// Whether the tool is deploy-capable.
    pub fn is_artifact_tool(&self, name: &str) -> bool {
        self.artifact_tools
            .iter()
            .any(|pattern| pattern_match(pattern, name).is_some())
    }

    /// The overwrite spec for a file-writing tool.
    pub fn overwrite_spec(&self, name: &str) -> Option<&OverwriteSpec> {
        best_match(self.overwrite_tools.iter(), name)
    }

    /// The first adapter marker found anywhere in `arguments` (keys or string
    /// values, recursively).
    pub fn adapter_marker_in(&self, arguments: &Map<String, Value>) -> Option<&str> {
        let mut texts = Vec::new();
        collect_strings(&Value::Object(arguments.clone()), &mut texts);
        self.adapter_markers
            .iter()
            .find(|marker| texts.iter().any(|text| text.contains(marker.as_str())))
            .map(String::as_str)
    }
}

fn collect_strings(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(text) => out.push(text.clone()),
        Value::Array(items) => items.iter().for_each(|item| collect_strings(item, out)),
        Value::Object(map) => {
            for (key, item) in map {
                out.push(key.clone());
                collect_strings(item, out);
            }
        }
        _ => {}
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
    /// The Chief-of-Staff log: announcements plus the write-ahead spend
    /// ledger. Must lie outside every root she can write through file_io/edit.
    pub announcements: PathBuf,
    /// The spend ceiling she is given.
    pub budget: Money,
}

impl ChiefOfStaffConfig {
    /// Refuse a log path inside any of the akh's writable sandbox roots.
    pub fn check_log_outside(&self, write_roots: &[PathBuf]) -> CosResult<()> {
        let log = crate::fs::canonicalize_existing_prefix(&absolute(&self.announcements));
        for root in write_roots {
            let root = crate::fs::canonicalize_existing_prefix(&absolute(root));
            if log.starts_with(&root) {
                return Err(CosError::LogInsideSandbox {
                    log: log.display().to_string(),
                    root: root.display().to_string(),
                });
            }
        }
        Ok(())
    }
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
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

/// Where a commit stands relative to the main ref.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ancestry {
    /// On main.
    Ancestor,
    /// A commit, but not on main.
    NotAncestor,
    /// Not a commit in the checkout (not fetched, or not a commit at all).
    NotACommit,
}

/// Decides where a commit stands relative to the main ref.
pub trait AncestryOracle: Send + Sync {
    /// Classify `sha` against `main_ref`.
    fn ancestry(&self, sha: &CommitSha, main_ref: &str) -> CosResult<Ancestry>;
}

/// [`AncestryOracle`] backed by git in a checkout.
///
/// It reads the local main ref and does not fetch. A stale ref can only
/// refuse a deploy that would have been allowed, never allow one from a
/// branch.
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

    fn git(&self, args: &[&str]) -> CosResult<std::process::Output> {
        std::process::Command::new("git")
            .arg("-C")
            .arg(&self.repository)
            .args(args)
            .output()
            .map_err(|source| CosError::AncestryIo {
                repository: self.repository.display().to_string(),
                source,
            })
    }
}

impl AncestryOracle for GitAncestry {
    fn ancestry(&self, sha: &CommitSha, main_ref: &str) -> CosResult<Ancestry> {
        let unknown = |output: &std::process::Output| CosError::AncestryUnknown {
            sha: sha.to_string(),
            main_ref: main_ref.to_string(),
            detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        };
        let object = format!("{sha}^{{commit}}");
        let verify = self.git(&["rev-parse", "--verify", "--quiet", &object])?;
        match verify.status.code() {
            Some(0) => {}
            Some(1) => return Ok(Ancestry::NotACommit),
            _ => return Err(unknown(&verify)),
        }
        let output = self.git(&["merge-base", "--is-ancestor", sha.as_str(), main_ref])?;
        match output.status.code() {
            Some(0) => Ok(Ancestry::Ancestor),
            Some(1) => Ok(Ancestry::NotAncestor),
            _ => Err(unknown(&output)),
        }
    }
}

fn ancestry_evidence(sha: &CommitSha, main_ref: &str, now: UnixSeconds) -> (RuleCheck, Receipt) {
    let command = format!("git merge-base --is-ancestor {sha} {main_ref}");
    (
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
    )
}

/// Check the declared deploy/restart SHA. It must be a commit on main.
pub fn check_main_only(
    oracle: &dyn AncestryOracle,
    sha: &CommitSha,
    main_ref: &str,
    now: UnixSeconds,
) -> CosResult<(RuleCheck, Receipt)> {
    match oracle.ancestry(sha, main_ref)? {
        Ancestry::Ancestor => Ok(ancestry_evidence(sha, main_ref, now)),
        Ancestry::NotAncestor => Err(CosError::NonMainArtifact {
            sha: sha.to_string(),
            main_ref: main_ref.to_string(),
            found_in: RefSource::DeclaredAction,
        }),
        Ancestry::NotACommit => Err(CosError::AncestryUnknown {
            sha: sha.to_string(),
            main_ref: main_ref.to_string(),
            detail: "not a commit in the configured checkout".into(),
        }),
    }
}

static UUID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
        .expect("fixed UUID pattern")
});

/// Commit-like references in the arguments: runs of 7..40 lowercase hex
/// digits with at least one digit and one letter, bounded by non-alphanumeric
/// characters (so `akhomed-main-1a2b3c4` and `:main-1a2b3c4d` yield their
/// SHA). Full UUIDs are ignored.
pub fn commit_refs_in(arguments: &Map<String, Value>) -> BTreeSet<CommitSha> {
    let mut texts = Vec::new();
    collect_strings(&Value::Object(arguments.clone()), &mut texts);
    texts
        .iter()
        .flat_map(|text| {
            UUID.replace_all(text, " ")
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|token| {
                    (7..=40).contains(&token.len())
                        && token
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                        && token.bytes().any(|b| b.is_ascii_digit())
                        && token.bytes().any(|b| b.is_ascii_lowercase())
                })
                .filter_map(|token| CommitSha::try_from(token.to_string()).ok())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Check every commit referenced in a deploy-capable tool's arguments. A
/// commit that is not on main is refused; a token that is not a commit in the
/// checkout is not a reference and is skipped.
pub fn check_argument_refs(
    oracle: &dyn AncestryOracle,
    arguments: &Map<String, Value>,
    main_ref: &str,
    now: UnixSeconds,
) -> CosResult<Vec<(RuleCheck, Receipt)>> {
    let refs = commit_refs_in(arguments);
    if refs.len() > MAX_ARGUMENT_REFS {
        return Err(CosError::InvalidValue {
            field: "tool arguments",
            message: format!(
                "{} commit-like references (limit {MAX_ARGUMENT_REFS}); split the call",
                refs.len()
            ),
        });
    }
    let mut checks = Vec::new();
    for sha in &refs {
        match oracle.ancestry(sha, main_ref)? {
            Ancestry::Ancestor => checks.push(ancestry_evidence(sha, main_ref, now)),
            Ancestry::NotAncestor => {
                return Err(CosError::NonMainArtifact {
                    sha: sha.to_string(),
                    main_ref: main_ref.to_string(),
                    found_in: RefSource::ToolArguments,
                });
            }
            Ancestry::NotACommit => {}
        }
    }
    Ok(checks)
}

// ── Standing rule: verified target ──────────────────────────────────────────

/// Computes the digest of a local target (file or directory tree).
pub trait DigestOracle: Send + Sync {
    /// `sha256:<hex>` of `target`, or `None` if it cannot be read.
    fn digest(&self, target: &Path) -> Option<String>;
}

/// [`DigestOracle`] over the local filesystem. A file digests its bytes. A
/// directory digests its sorted entries (relative path + entry digest). A
/// symlink digests its link text and is never followed.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalDigest;

impl LocalDigest {
    fn hash_path(path: &Path, hasher: &mut Sha256) -> std::io::Result<()> {
        let meta = std::fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() {
            hasher.update(b"L");
            hasher.update(std::fs::read_link(path)?.as_os_str().as_encoded_bytes());
        } else if meta.is_dir() {
            hasher.update(b"D");
            let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?
                .map(|entry| entry.map(|e| e.path()))
                .collect::<std::io::Result<_>>()?;
            entries.sort();
            for entry in entries {
                hasher.update(entry.file_name().unwrap_or_default().as_encoded_bytes());
                hasher.update([0]);
                Self::hash_path(&entry, hasher)?;
            }
        } else {
            hasher.update(b"F");
            let mut file = std::fs::File::open(path)?;
            let mut buffer = [0u8; 64 * 1024];
            loop {
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
        }
        Ok(())
    }
}

impl DigestOracle for LocalDigest {
    fn digest(&self, target: &Path) -> Option<String> {
        let mut hasher = Sha256::new();
        Self::hash_path(target, &mut hasher).ok()?;
        let hex: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        Some(format!("sha256:{hex}"))
    }
}

/// Normalise an observed digest (`sha256:ABC…` or bare hex) for comparison.
fn normalise_digest(digest: &str) -> String {
    let lower = digest.trim().to_ascii_lowercase();
    match lower.strip_prefix("sha256:") {
        Some(_) => lower,
        None => format!("sha256:{lower}"),
    }
}

/// The verified-target rule. The runtime digests `target` itself and compares
/// it with `observed_digest`. For an irreversible action it also digests
/// `copy`, which must be a different path with the same digest.
pub fn check_verified_target(
    digests: &dyn DigestOracle,
    operation: &'static str,
    target: &Path,
    observed_digest: &str,
    reversibility: Reversibility,
    copy: Option<&Path>,
    now: UnixSeconds,
) -> CosResult<(RuleCheck, Vec<Receipt>)> {
    let refuse = |gap| CosError::UnverifiedDeletion {
        operation,
        target: target.display().to_string(),
        gap,
    };
    let receipt = |path: &Path, digest: &str| Receipt {
        kind: ReceiptKind::Digest,
        reference: path.display().to_string(),
        observed_at: now,
        digest: Some(digest.to_string()),
    };
    let actual = digests
        .digest(target)
        .ok_or_else(|| refuse(DeletionGap::TargetUnreadable))?;
    if normalise_digest(observed_digest) != actual {
        return Err(refuse(DeletionGap::TargetChanged));
    }
    let mut receipts = vec![receipt(target, &actual)];
    let evidence = match (reversibility, copy) {
        (Reversibility::Irreversible, None) => return Err(refuse(DeletionGap::NoVerifiedCopy)),
        (_, Some(copy)) => {
            let same = crate::fs::canonicalize_existing_prefix(copy)
                == crate::fs::canonicalize_existing_prefix(target);
            if same {
                return Err(refuse(DeletionGap::CopyIsTarget));
            }
            let copy_digest = digests
                .digest(copy)
                .ok_or_else(|| refuse(DeletionGap::CopyUnreadable))?;
            if copy_digest != actual {
                return Err(refuse(DeletionGap::CopyDigestMismatch));
            }
            receipts.push(receipt(copy, &copy_digest));
            format!(
                "runtime digest of {} is {actual}, matching the observed digest; copy at {} verified",
                target.display(),
                copy.display()
            )
        }
        (Reversibility::Reversible, None) => format!(
            "runtime digest of {} is {actual}, matching the observed digest; reversible",
            target.display()
        ),
    };
    Ok((
        RuleCheck {
            rule: StandingRule::VerifiedDeletion,
            evidence,
        },
        receipts,
    ))
}
