//! Lilith C1 exit criteria (ADR 116, plan slice C1) and the PR #436 review
//! fixes. Covered here:
//!
//! - schema tests;
//! - a T3 action taken and announced with its arguments, receipts and rollback
//!   path;
//! - a non-main deploy refused, whether the SHA is declared or only appears
//!   in the arguments;
//! - a T2 post carrying receipts;
//! - adapter activation refused, by tool name, by declared action or by an
//!   argument marker;
//! - runtime-computed digests for deletions and overwrites;
//! - the write-ahead spend ledger, crash tolerance, panics and concurrency;
//! - the fail-closed fallback;
//! - the advertised declaration parameter.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use miette::Diagnostic;
use serde_json::{Map, Value, json};

use crate::agent::error::AgentResult;
use crate::agent::permission::{CallOutcome, PermissionMode, ToolCallGuard};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolRegistry, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolSource,
};
use crate::engine::{Engine, EngineConfig};
use crate::vsa::Dimension;

use super::contracts::*;
use super::ledger::{SpendReservation, SpendSettlement};
use super::*;

// ── Fixtures ────────────────────────────────────────────────────────────────

const MAIN_SHA: &str = "4f2a9c1e0b7d";
const BRANCH_SHA: &str = "deadbeef1234";
const NOW: UnixSeconds = 1_790_000_000;

fn fixed_clock() -> UnixSeconds {
    NOW
}

fn skill_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("skills/chief-of-staff")
}

fn schema(name: &str) -> Value {
    let path = skill_dir()
        .join("schemas")
        .join(format!("{name}.schema.json"));
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn validator(name: &str) -> jsonschema::Validator {
    jsonschema::validator_for(&schema(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn assert_valid_against(validator: &jsonschema::Validator, name: &str, instance: &Value) {
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:?}\n{instance:#}");
}

fn assert_valid(name: &str, instance: &Value) {
    assert_valid_against(&validator(name), name, instance);
}

/// Schema name for a contract id (`lilith/cos-briefing@1` → `cos-briefing`).
fn schema_name(contract: &str) -> &str {
    contract
        .strip_prefix("lilith/")
        .and_then(|rest| rest.strip_suffix("@1"))
        .unwrap()
}

fn receipt(kind: ReceiptKind, reference: &str) -> Receipt {
    Receipt {
        kind,
        reference: reference.into(),
        observed_at: NOW,
        digest: None,
    }
}

fn receipts(kind: ReceiptKind, reference: &str) -> Receipts {
    Receipts::try_from(vec![receipt(kind, reference)]).unwrap()
}

fn lang(tag: &str) -> LanguageTag {
    LanguageTag::try_from(tag.to_string()).unwrap()
}

fn sha(value: &str) -> CommitSha {
    CommitSha::try_from(value.to_string()).unwrap()
}

fn usd(minor_units: u64) -> Money {
    Money {
        minor_units,
        currency: CurrencyCode::try_from("USD".to_string()).unwrap(),
    }
}

fn manifest(name: &str, effect: ToolEffect) -> ToolManifest {
    ToolManifest {
        name: name.into(),
        description: name.into(),
        parameters: vec![],
        danger: DangerInfo {
            effect,
            level: DangerLevel::Dangerous,
            capabilities: HashSet::from([Capability::ProcessExec]),
            description: name.into(),
            shadow_triggers: vec![],
        },
        source: ToolSource::Native,
    }
}

fn args(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap()
}

fn no_args() -> Map<String, Value> {
    Map::new()
}

fn shipped_profile() -> CosPermissionProfile {
    CosPermissionProfile::load(&skill_dir().join("permission-profile.json")).unwrap()
}

/// A repository with one main commit and one branch commit; every other
/// token is not a commit.
struct FakeRepo;

impl AncestryOracle for FakeRepo {
    fn ancestry(&self, sha: &CommitSha, _main_ref: &str) -> CosResult<Ancestry> {
        Ok(match sha.as_str() {
            MAIN_SHA => Ancestry::Ancestor,
            BRANCH_SHA => Ancestry::NotAncestor,
            _ => Ancestry::NotACommit,
        })
    }
}

fn guard_on(profile: CosPermissionProfile, budget: u64, log: Arc<dyn CosLog>) -> CosGuard {
    CosGuard::new(
        profile,
        SpendLedger::new(usd(budget)),
        Box::new(FakeRepo),
        log,
    )
    .with_clock(fixed_clock)
}

fn guard_with(profile: CosPermissionProfile, budget: u64) -> (Arc<CosGuard>, Arc<MemoryLog>) {
    let log = Arc::new(MemoryLog::default());
    (Arc::new(guard_on(profile, budget, log.clone())), log)
}

fn guard() -> (Arc<CosGuard>, Arc<MemoryLog>) {
    guard_with(shipped_profile(), 5_000)
}

/// A tool that records the inputs it actually received, or panics.
struct RecordingTool {
    name: &'static str,
    effect: ToolEffect,
    panics: bool,
    calls: Arc<Mutex<Vec<ToolInput>>>,
}

impl Tool for RecordingTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: self.name.into(),
            description: self.name.into(),
            parameters: vec![],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        self.calls.lock().unwrap().push(input);
        assert!(!self.panics, "tool exploded mid-call");
        Ok(ToolOutput::ok("done"))
    }

    fn manifest(&self) -> ToolManifest {
        manifest(self.name, self.effect)
    }
}

fn registry_with(
    guard: Arc<dyn ToolCallGuard>,
    tools: &[(&'static str, ToolEffect, bool)],
) -> (ToolRegistry, Arc<Mutex<Vec<ToolInput>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let registry = ToolRegistry::new();
    for (name, effect, panics) in tools {
        registry.register(Box::new(RecordingTool {
            name,
            effect: *effect,
            panics: *panics,
            calls: calls.clone(),
        }));
    }
    registry.set_tool_call_guard(guard);
    (registry, calls)
}

fn engine() -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap()
}

fn operator(action: Value) -> Value {
    json!({
        "tier": "operator-class",
        "action": action,
        "receipts": [{"kind": "run-record", "reference": "ci/job/8812", "observed_at": NOW}],
        "reason": "CI green on the fix; the running binary predates it",
        "rollback": {"kind": "revert", "steps": "redeploy akhomed-main-1a2b3c4 and restart"}
    })
}

fn deploy_declaration(artifact_sha: &str) -> Value {
    operator(json!({"kind": "deploy", "artifact_sha": artifact_sha, "target": "akhomed@archibald"}))
}

fn declared(value: Value) -> Option<Declaration> {
    Some(serde_json::from_value(value).unwrap())
}

fn spend(minor_units: u64) -> Declaration {
    Declaration::OperatorClass {
        action: OperatorAction::Spend {
            amount: usd(minor_units),
            purpose: "H100 hour".into(),
        },
        receipts: vec![receipt(ReceiptKind::RunRecord, "runpod/pricing")],
        reason: "C6 smoke".into(),
        rollback: RollbackPath::Irreversible {
            mitigation: "auto-terminate".into(),
        },
    }
}

fn digest_of(path: &Path) -> String {
    LocalDigest.digest(path).unwrap()
}

// ── Schema tests ────────────────────────────────────────────────────────────

fn sample_documents() -> Vec<CosDocument> {
    let item = BriefingItem {
        entity: EntityRef {
            kind: EntityKind::WorkItem,
            id: "01a0e270-0288".into(),
        },
        summary: "parkiert: CI-Timeout".into(),
        receipts: receipts(ReceiptKind::WorkItem, "01a0e270-0288"),
    };
    let deploy = OperatorAction::Deploy {
        artifact_sha: sha(MAIN_SHA),
        target: "akhomed@archibald".into(),
    };
    let rollback = RollbackPath::Revert {
        steps: "redeploy the previous main SHA".into(),
    };
    vec![
        CosDocument::Briefing(CosBriefing {
            language: lang("de-CH"),
            generated_at: NOW,
            done: vec![],
            in_flight: vec![item.clone()],
            blocked: vec![item.clone()],
            next: vec![],
        }),
        CosDocument::ActionItem(CosActionItem {
            language: lang("de"),
            action: "Runpod-Guthaben aufladen".into(),
            location: "runpod.io console (not in this repo)".into(),
            trigger: Trigger::Deadline { due_at: NOW + 3600 },
            cost_of_skipping: "C6 smoke run cannot start".into(),
            entities: vec![],
            receipts: receipts(ReceiptKind::RunRecord, "runpod/billing"),
        }),
        CosDocument::Announcement(CosAnnouncement {
            akh: "lilith".into(),
            announced_at: NOW,
            tool: "shell_exec".into(),
            arguments: args(json!({"command": "deploy/akhomed.sh"})),
            action: deploy.clone(),
            receipts: receipts(ReceiptKind::AncestryCheck, "git merge-base --is-ancestor"),
            reason: "fix landed".into(),
            rollback: rollback.clone(),
            rule_checks: vec![RuleCheck {
                rule: StandingRule::MainOnlyDeploy,
                evidence: "exit 0".into(),
            }],
            outcome: ActionOutcome::Unknown {
                detail: "unwound".into(),
            },
        }),
        CosDocument::Commitment(CosCommitment {
            id: "c-1".into(),
            language: lang("de"),
            owner: CommitmentOwner::Delegate,
            counterparty: Some("tecton".into()),
            statement: "PR #420 reviewed by Friday".into(),
            due_at: Some(NOW + 86_400),
            closing_evidence: "review posted on PR #420".into(),
            status: CommitmentStatus::Closed {
                evidence: receipts(ReceiptKind::PullRequest, "420"),
            },
            entities: vec![EntityRef {
                kind: EntityKind::PullRequest,
                id: "420".into(),
            }],
            receipts: receipts(ReceiptKind::BoardMessage, "board/msg/77"),
        }),
        CosDocument::Guidance(CosGuidance {
            language: lang("en"),
            audience: GuidanceAudience::Implementer,
            subject: EntityRef {
                kind: EntityKind::PullRequest,
                id: "431".into(),
            },
            subsystem: IllumosSubsystem::Smf,
            severity: GuidanceSeverity::Blocker,
            finding: "the manifest starts the daemon from /etc/init.d".into(),
            recommendation: "ship an SMF manifest with a method script".into(),
            citations: vec![Citation {
                kind: CitationKind::ManPage,
                reference: "smf_method(7)".into(),
            }],
            receipts: receipts(ReceiptKind::PullRequest, "431"),
        }),
        CosDocument::Proposal(CosProposal {
            language: lang("de"),
            motive: ProposalMotive::OutsideGrant,
            action: OperatorAction::Delete {
                target: "/srv/runs/old".into(),
                reversibility: Reversibility::Irreversible,
                observed_digest: "sha256:aa".into(),
                copy: Some("/nfs/runs/old".into()),
            },
            receipts: receipts(ReceiptKind::RunRecord, "runs/c7/report"),
            reason: "volume retired".into(),
            rollback: RollbackPath::RestoreFromCopy {
                location: "/nfs/runs/old".into(),
            },
        }),
    ]
}

#[test]
fn every_schema_compiles() {
    for name in [
        "cos-briefing",
        "cos-action-item",
        "cos-announcement",
        "cos-commitment",
        "cos-guidance",
        "cos-proposal",
        "cos-declaration",
        "cos-permission-profile",
    ] {
        validator(name);
    }
}

#[test]
fn typed_documents_validate_against_their_schemas_and_round_trip() {
    for document in sample_documents() {
        let wire = serde_json::to_value(&document).unwrap();
        assert_eq!(wire["contract"], document.contract_id());
        assert_valid(schema_name(document.contract_id()), &wire);
        let back: CosDocument = serde_json::from_value(wire).unwrap();
        assert_eq!(back, document);
    }
    // Every declaration shape the guard accepts is valid against the
    // advertised declaration schema.
    for declaration in [
        deploy_declaration(MAIN_SHA),
        serde_json::to_value(spend(10)).unwrap(),
        json!({"tier": "coordination", "receipts": [{"kind": "work-item", "reference": "x", "observed_at": 1}]}),
    ] {
        serde_json::from_value::<Declaration>(declaration.clone()).unwrap();
        assert_valid("cos-declaration", &declaration);
    }
}

#[test]
fn schemas_and_types_reject_missing_standing_fields() {
    let documents = sample_documents();
    let wire = |index: usize| serde_json::to_value(&documents[index]).unwrap();

    let mut action_item = wire(1);
    action_item
        .as_object_mut()
        .unwrap()
        .remove("cost_of_skipping");
    let mut briefing = wire(0);
    briefing["blocked"][0]["receipts"] = json!([]);
    let mut announcement = wire(2);
    announcement.as_object_mut().unwrap().remove("rollback");
    let mut no_arguments = wire(2);
    no_arguments.as_object_mut().unwrap().remove("arguments");
    let mut not_a_sha = wire(2);
    not_a_sha["action"]["artifact_sha"] = json!("--upload-pack=evil");
    let mut uncited = wire(4);
    uncited["citations"] = json!([]);
    let mut language = wire(5);
    language["language"] = json!("German");
    let mut unobserved = wire(5);
    unobserved["action"]
        .as_object_mut()
        .unwrap()
        .remove("observed_digest");

    for (name, instance) in [
        ("cos-action-item", action_item),
        ("cos-briefing", briefing),
        ("cos-announcement", announcement),
        ("cos-announcement", no_arguments),
        ("cos-announcement", not_a_sha),
        ("cos-guidance", uncited),
        ("cos-proposal", language),
        ("cos-proposal", unobserved),
    ] {
        assert!(
            !validator(name).is_valid(&instance),
            "{name} schema accepted {instance:#}"
        );
        // The typed layer agrees, except for the one invariant only the
        // schema expresses (guidance citations: minItems in the schema).
        let typed = serde_json::from_value::<CosDocument>(instance.clone());
        if name != "cos-guidance" {
            assert!(typed.is_err(), "{name} type accepted {instance:#}");
        }
    }
}

#[test]
fn shipped_profile_matches_its_schema_and_grants_every_tier() {
    let text = std::fs::read_to_string(skill_dir().join("permission-profile.json")).unwrap();
    assert_valid(
        "cos-permission-profile",
        &serde_json::from_str::<Value>(&text).unwrap(),
    );
    let profile = shipped_profile();
    assert_eq!(profile.akh, "lilith");
    assert_eq!(profile.main_ref, "origin/main");
    for tier in AutonomyTier::ALL {
        assert_eq!(profile.grant(tier), TierGrant::Autonomous, "{tier}");
    }
}

#[test]
fn skill_pack_loads_through_the_skill_loader() {
    let loaded = crate::skills::procedures::SkillProcedures::load(&skill_dir()).unwrap();
    assert_eq!(loaded.label(), "chief-of-staff@1.0.0");
    let triples: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(skill_dir().join("triples.json")).unwrap())
            .unwrap();
    assert!(!triples.is_empty());
}

// ── Classification ──────────────────────────────────────────────────────────

#[test]
fn classification_uses_exact_names_then_longest_prefix_then_fails_closed() {
    let profile = shipped_profile();
    let class = |name: &str, effect| profile.classify(&manifest(name, effect));
    assert_eq!(class("kg_query", ToolEffect::ReadOnly), ToolClass::T0Read);
    assert_eq!(
        class("user_interact", ToolEffect::Mutating),
        ToolClass::T1OwnState,
        "asking the operator needs no receipts"
    );
    assert_eq!(
        class("mcp__anima__get_work_item", ToolEffect::Mutating),
        ToolClass::T0Read
    );
    assert_eq!(
        class("mcp__anima__post_message", ToolEffect::Mutating),
        ToolClass::T2Coordination
    );
    assert_eq!(
        class("mcp__anima__approve_and_merge", ToolEffect::Mutating),
        ToolClass::T3OperatorClass
    );
    assert_eq!(
        class("lilith_adapter_activate", ToolEffect::Mutating),
        ToolClass::OutsideGrant
    );
    assert_eq!(
        class("never_seen_reader", ToolEffect::ReadOnly),
        ToolClass::T0Read
    );
    assert_eq!(
        class("never_seen_writer", ToolEffect::Mutating),
        ToolClass::T3OperatorClass
    );
}

#[test]
fn read_only_diagnostics_are_t0_and_the_operator_class_writes_beside_them_stay_t3() {
    // Dojo run evaluation D-D (2026-09-30): these three were filed by the `mcp__anima__*` catch-all
    // as operator-class writes, so reading them counted as an unannounced operator action.
    let profile = shipped_profile();
    let class = |name: &str| profile.classify(&manifest(name, ToolEffect::Mutating));
    for read in [
        "mcp__anima__stranded_work_items",
        "mcp__anima__lane_occupancy",
        "mcp__anima__overdue_ci_dispatches",
        "mcp__anima__session_lease_would_reclaim",
        "mcp__anima__witness_gap_distribution",
        "mcp__anima__resolve_project_by_git_remote",
        "mcp__runpod__list-pods",
        "mcp__runpod__get-pod",
        "mcp__runpod__stream-pod-logs",
    ] {
        assert_eq!(class(read), ToolClass::T0Read, "{read}");
    }
    for write in [
        "mcp__anima__unpark_work_item",
        "mcp__anima__run_scheduler_tick",
        "mcp__anima__seshat_wish",
        "mcp__runpod__create-pod",
        "mcp__runpod__delete-pod",
        "mcp__runpod__pod-action",
    ] {
        assert_eq!(class(write), ToolClass::T3OperatorClass, "{write}");
    }
}

// ── T3 taken and announced ──────────────────────────────────────────────────

#[test]
fn t3_action_is_taken_and_announced_with_arguments_receipts_and_rollback_path() {
    let (guard, log) = guard();
    let (registry, calls) = registry_with(guard, &[("shell_exec", ToolEffect::Mutating, false)]);
    let secret = format!("ghp_{}", "x".repeat(36));
    let command = format!("GH_TOKEN={secret} deploy/akhomed.sh akhomed-main-{MAIN_SHA}");
    let input = ToolInput::new()
        .with_param("command", command.clone())
        .with_value(DECLARATION_PARAM, deploy_declaration(MAIN_SHA));

    let out = registry.execute("shell_exec", input, &engine()).unwrap();
    assert!(out.success, "{}", out.result);

    // The action ran with its real input, and the tool never saw the declaration.
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].get_value(DECLARATION_PARAM).is_none());
    assert_eq!(calls[0].get("command"), Some(command.as_str()));

    let announced = log.announcements();
    assert_eq!(announced.len(), 1);
    let announcement = &announced[0];
    assert_eq!(announcement.akh, "lilith");
    assert_eq!(announcement.tool, "shell_exec");
    assert!(announcement.outcome.succeeded());
    assert!(matches!(announcement.rollback, RollbackPath::Revert { .. }));

    // (2a) The real arguments are announced, declaration stripped, the
    // credential redacted.
    let announced_command = announcement.arguments["command"].as_str().unwrap();
    assert!(announced_command.contains("deploy/akhomed.sh"));
    assert!(!announced_command.contains(&secret), "{announced_command}");
    assert!(!announcement.arguments.contains_key(DECLARATION_PARAM));

    // Her receipt plus one quoted ancestry check (deduplicated between the
    // declared SHA and the same SHA found in the arguments).
    let kinds: Vec<ReceiptKind> = announcement
        .receipts
        .as_slice()
        .iter()
        .map(|r| r.kind)
        .collect();
    assert_eq!(
        kinds,
        vec![ReceiptKind::RunRecord, ReceiptKind::AncestryCheck]
    );
    assert_eq!(
        announcement.receipts.as_slice()[1].reference,
        format!("git merge-base --is-ancestor {MAIN_SHA} origin/main")
    );
    assert!(
        announcement
            .rule_checks
            .iter()
            .any(|c| c.rule == StandingRule::MainOnlyDeploy)
    );

    let wire = serde_json::to_value(CosDocument::Announcement(announcement.clone())).unwrap();
    assert_valid("cos-announcement", &wire);
}

// ── Main-only deploy ────────────────────────────────────────────────────────

#[test]
fn non_main_deploy_is_refused_by_the_rule_check_with_a_diagnostic() {
    let (guard, log) = guard();
    let err = guard
        .admit(
            &manifest("shell_exec", ToolEffect::Mutating),
            declared(deploy_declaration(BRANCH_SHA)),
            &no_args(),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            CosError::NonMainArtifact {
                found_in: RefSource::DeclaredAction,
                ..
            }
        ),
        "{err:?}"
    );
    assert_eq!(err.standing_rule(), Some(StandingRule::MainOnlyDeploy));
    assert_eq!(
        err.code().unwrap().to_string(),
        "akh::cos::non_main_artifact"
    );
    assert!(err.help().unwrap().to_string().contains("origin/main"));

    let (registry, calls) = registry_with(guard, &[("shell_exec", ToolEffect::Mutating, false)]);
    let input = ToolInput::new().with_value(DECLARATION_PARAM, deploy_declaration(BRANCH_SHA));
    let out = registry.execute("shell_exec", input, &engine()).unwrap();
    assert!(!out.success);
    assert!(
        out.result.contains("akh::cos::non_main_artifact"),
        "{}",
        out.result
    );
    assert!(calls.lock().unwrap().is_empty());
    assert!(log.announcements().is_empty());
}

#[test]
fn a_mis_declared_action_cannot_carry_a_branch_sha_through_a_deploy_capable_tool() {
    let (guard, _) = guard();
    let shell = manifest("shell_exec", ToolEffect::Mutating);
    let merge = || declared(operator(json!({"kind": "merge", "change": "PR #436"})));

    // Declared as a merge, but the command deploys a branch build.
    let err = guard
        .admit(
            &shell,
            merge(),
            &args(json!({"command": format!("kubectl set image deploy/akhomed akhomed=code.aopc.cloud/akhomed:{BRANCH_SHA}")})),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            CosError::NonMainArtifact {
                found_in: RefSource::ToolArguments,
                ..
            }
        ),
        "{err:?}"
    );

    // Hex-looking tokens that are not commits (UUIDs, Anima short ids) and
    // main SHAs pass; the main SHA is quoted as evidence.
    let admitted = guard
        .admit(
            &shell,
            merge(),
            &args(json!({"command": format!(
                "deploy {MAIN_SHA} for 01a0e270-0288-7ac1-b297-a45375d4e6e6 and 01a0e270-0288"
            )})),
        )
        .unwrap();
    let Admission::OperatorClass(authorized) = admitted else {
        panic!("expected an operator-class admission");
    };
    assert!(
        authorized
            .receipts
            .as_slice()
            .iter()
            .any(|r| r.reference.contains(MAIN_SHA))
    );
}

#[test]
fn git_ancestry_decides_against_a_real_repository() {
    let git = |dir: &Path, args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
    };
    let dir = tempfile::tempdir().unwrap();
    let Ok(init) = git(dir.path(), &["init", "-q"]) else {
        eprintln!("git not installed; GitAncestry exercised through FakeRepo only");
        return;
    };
    assert!(init.status.success());
    let head = |dir: &Path| {
        let out = git(dir, &["rev-parse", "HEAD"]).unwrap();
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    };
    let commit = |message: &str| {
        assert!(
            git(
                dir.path(),
                &["commit", "-q", "--allow-empty", "-m", message]
            )
            .unwrap()
            .status
            .success()
        );
    };
    commit("a");
    let on_main = head(dir.path());
    assert!(
        git(
            dir.path(),
            &["update-ref", "refs/remotes/origin/main", &on_main]
        )
        .unwrap()
        .status
        .success()
    );
    commit("b");
    let on_branch = head(dir.path());

    let oracle = GitAncestry::new(dir.path());
    assert_eq!(
        oracle.ancestry(&sha(&on_main), "origin/main").unwrap(),
        Ancestry::Ancestor
    );
    assert_eq!(
        oracle.ancestry(&sha(&on_branch), "origin/main").unwrap(),
        Ancestry::NotAncestor
    );
    assert_eq!(
        oracle
            .ancestry(&sha("0123456789abcdef0123"), "origin/main")
            .unwrap(),
        Ancestry::NotACommit
    );
    let outside = tempfile::tempdir().unwrap();
    assert!(matches!(
        GitAncestry::new(outside.path()).ancestry(&sha(&on_main), "origin/main"),
        Err(CosError::AncestryUnknown { .. })
    ));
}

// ── T2 receipts, T0/T1 ──────────────────────────────────────────────────────

#[test]
fn t2_post_carries_receipts() {
    let (guard, log) = guard();
    let post = manifest("mcp__anima__post_message", ToolEffect::Mutating);

    let admission = guard
        .admit(
            &post,
            Some(Declaration::Coordination {
                receipts: vec![receipt(ReceiptKind::WorkItem, "01a0e270-0288")],
            }),
            &no_args(),
        )
        .unwrap();
    let settled = guard
        .settle(
            admission,
            ActionOutcome::Succeeded {
                detail: "posted".into(),
            },
        )
        .unwrap();
    let Settlement::Coordinated { tool, receipts } = settled else {
        panic!("expected a coordination settlement, got {settled:?}");
    };
    assert_eq!(tool, "mcp__anima__post_message");
    assert_eq!(receipts.as_slice()[0].reference, "01a0e270-0288");
    assert!(log.announcements().is_empty());

    let no_receipts = guard
        .admit(
            &post,
            Some(Declaration::Coordination { receipts: vec![] }),
            &no_args(),
        )
        .unwrap_err();
    assert!(matches!(no_receipts, CosError::MissingReceipts { .. }));
    let undeclared = guard.admit(&post, None, &no_args()).unwrap_err();
    assert!(matches!(
        undeclared,
        CosError::MissingDeclaration {
            tier: AutonomyTier::Coordination,
            ..
        }
    ));
}

#[test]
fn t0_and_t1_run_without_a_declaration() {
    let (guard, _) = guard();
    for (name, tier) in [
        ("kg_query", AutonomyTier::Read),
        ("memory_write", AutonomyTier::OwnState),
        ("user_interact", AutonomyTier::OwnState),
    ] {
        let admission = guard
            .admit(&manifest(name, ToolEffect::Mutating), None, &no_args())
            .unwrap();
        assert_eq!(
            admission,
            Admission::Autonomous {
                tool: name.into(),
                tier
            }
        );
    }
}

// ── Adapter activation ──────────────────────────────────────────────────────

#[test]
fn adapter_activation_is_refused_by_tool_action_and_argument() {
    let (guard, log) = guard();

    let by_tool = guard
        .admit(
            &manifest("lilith_adapter_activate", ToolEffect::Mutating),
            None,
            &no_args(),
        )
        .unwrap_err();
    assert!(matches!(by_tool, CosError::OutsideGrant { .. }));

    let by_action = operator(json!({"kind": "activate-adapter", "adapter": "lilith-cos-r1"}));
    // (2c) Declared as something harmless, but the arguments activate an
    // adapter on the serving endpoint.
    let by_argument = operator(json!({"kind": "outward", "destination": "vllm"}));
    let (registry, calls) = registry_with(guard, &[("shell_exec", ToolEffect::Mutating, false)]);
    for (declaration, command) in [
        (by_action, "true"),
        (
            by_argument,
            "curl -X POST http://serve:8000/v1/load_lora_adapter -d '{\"lora_name\":\"r1\"}'",
        ),
    ] {
        let input = ToolInput::new()
            .with_param("command", command)
            .with_value(DECLARATION_PARAM, declaration);
        let out = registry.execute("shell_exec", input, &engine()).unwrap();
        assert!(!out.success);
        assert!(
            out.result.contains("akh::cos::outside_grant"),
            "{}",
            out.result
        );
    }
    assert!(calls.lock().unwrap().is_empty());
    assert!(log.announcements().is_empty());
}

// ── Verified target (deletion and overwrite) ────────────────────────────────

#[test]
fn deletion_digests_are_computed_by_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("run.ckpt");
    let copy = dir.path().join("run.ckpt.bak");
    let other = dir.path().join("other.ckpt");
    std::fs::write(&target, b"weights").unwrap();
    std::fs::write(&copy, b"weights").unwrap();
    std::fs::write(&other, b"different").unwrap();
    let real = digest_of(&target);

    let (guard, _) = guard();
    let shell = manifest("shell_exec", ToolEffect::Mutating);
    let delete = |observed: &str, copy: Option<&Path>| {
        declared(operator(json!({
            "kind": "delete",
            "target": target.display().to_string(),
            "reversibility": "irreversible",
            "observed_digest": observed,
            "copy": copy.map(|c| c.display().to_string()),
        })))
    };
    let gap = |declaration| match guard.admit(&shell, declaration, &no_args()) {
        Err(CosError::UnverifiedDeletion { gap, .. }) => gap,
        other => panic!("expected a verified-deletion refusal, got {other:?}"),
    };

    // Two matching strings she supplies prove nothing: the runtime digests.
    assert_eq!(
        gap(delete("sha256:aa", Some(&copy))),
        DeletionGap::TargetChanged
    );
    assert_eq!(gap(delete(&real, None)), DeletionGap::NoVerifiedCopy);
    assert_eq!(
        gap(delete(&real, Some(&other))),
        DeletionGap::CopyDigestMismatch
    );
    assert_eq!(
        gap(delete(&real, Some(&dir.path().join("missing")))),
        DeletionGap::CopyUnreadable
    );
    assert_eq!(gap(delete(&real, Some(&target))), DeletionGap::CopyIsTarget);

    let admitted = guard.admit(&shell, delete(&real, Some(&copy)), &no_args());
    let Ok(Admission::OperatorClass(authorized)) = admitted else {
        panic!("expected an operator-class admission, got {admitted:?}");
    };
    let digests: Vec<&str> = authorized
        .receipts
        .as_slice()
        .iter()
        .filter(|r| r.kind == ReceiptKind::Digest)
        .filter_map(|r| r.digest.as_deref())
        .collect();
    assert_eq!(digests, vec![real.as_str(), real.as_str()]);
}

#[test]
fn file_io_and_edit_overwrites_of_existing_files_need_a_verified_target() {
    let scratch = tempfile::tempdir().unwrap();
    let notes = scratch.path().join("notes.md");
    std::fs::write(&notes, "v1").unwrap();
    let (guard, log) = {
        let log = Arc::new(MemoryLog::default());
        let guard = guard_on(shipped_profile(), 1_000, log.clone())
            .with_write_roots(vec![scratch.path().to_path_buf()]);
        (Arc::new(guard), log)
    };
    let file_io = manifest("file_io", ToolEffect::Mutating);
    let edit = manifest("edit", ToolEffect::Mutating);
    let write = |path: &str| args(json!({"action": "write", "path": path, "content": "v2"}));
    let overwrite = |target: &str, observed: &str| {
        declared(operator(json!({
            "kind": "overwrite",
            "target": target,
            "reversibility": "reversible",
            "observed_digest": observed,
        })))
    };
    let real = digest_of(&notes);

    // A new file is her own state: no declaration needed.
    assert!(matches!(
        guard.admit(&file_io, None, &write("fresh.md")).unwrap(),
        Admission::Autonomous { .. }
    ));
    // Reading an existing file is not a write.
    assert!(matches!(
        guard
            .admit(
                &file_io,
                None,
                &args(json!({"action": "read", "path": "notes.md"}))
            )
            .unwrap(),
        Admission::Autonomous { .. }
    ));

    let gap = |result: CosResult<Admission>| match result {
        Err(CosError::UnverifiedDeletion { gap, .. }) => gap,
        other => panic!("expected a verified-target refusal, got {other:?}"),
    };
    assert_eq!(
        gap(guard.admit(&file_io, None, &write("notes.md"))),
        DeletionGap::UndeclaredOverwrite
    );
    assert_eq!(
        gap(guard.admit(
            &edit,
            None,
            &args(json!({"path": "notes.md", "old_string": "v1", "new_string": "v2"}))
        )),
        DeletionGap::UndeclaredOverwrite
    );
    assert_eq!(
        gap(guard.admit(&file_io, overwrite("fresh.md", &real), &write("notes.md"))),
        DeletionGap::TargetMismatch
    );
    assert_eq!(
        gap(guard.admit(
            &file_io,
            overwrite("notes.md", "sha256:00"),
            &write("notes.md")
        )),
        DeletionGap::TargetChanged
    );

    // Declared, observed and matching: admitted as T3 and announced.
    let admission = guard
        .admit(&file_io, overwrite("notes.md", &real), &write("notes.md"))
        .unwrap();
    assert!(matches!(admission, Admission::OperatorClass(_)));
    guard
        .settle(
            admission,
            ActionOutcome::Succeeded {
                detail: "written".into(),
            },
        )
        .unwrap();
    assert_eq!(log.announcements().len(), 1);
}

// ── Budget: write-ahead, crash tolerance, panics, concurrency ───────────────

#[test]
fn spend_beyond_budget_is_refused_and_a_failed_spend_is_released() {
    let (guard, log) = guard_with(shipped_profile(), 1_000);
    let tool = manifest("mcp__runpod__create_pod", ToolEffect::Mutating);

    let too_much = guard
        .admit(&tool, Some(spend(1_001)), &no_args())
        .unwrap_err();
    assert!(matches!(
        too_much,
        CosError::BudgetExceeded {
            remaining: 1_000,
            ..
        }
    ));
    assert_eq!(too_much.standing_rule(), Some(StandingRule::WithinBudget));

    let admission = guard.admit(&tool, Some(spend(600)), &no_args()).unwrap();
    // (3a) The reservation is on the log before the call runs.
    assert!(matches!(
        log.records().as_slice(),
        [CosLogRecord::SpendReserved(SpendReservation { .. })]
    ));
    assert_eq!(guard.remaining_budget(), 400);
    guard
        .settle(
            admission,
            ActionOutcome::Failed {
                detail: "no stock".into(),
            },
        )
        .unwrap();
    assert_eq!(guard.remaining_budget(), 1_000, "failed spend is released");
    assert!(matches!(
        log.records().as_slice(),
        [
            CosLogRecord::SpendReserved(_),
            CosLogRecord::SpendSettled(SpendSettlement { spent: false, .. }),
            CosLogRecord::Announcement(_)
        ]
    ));
}

#[test]
fn unsettled_reservations_count_as_spent_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cos.jsonl");
    let (log, records) = JsonlLog::open(&path).unwrap();
    assert!(records.is_empty());
    let guard = guard_on(shipped_profile(), 1_000, Arc::new(log));
    let tool = manifest("mcp__runpod__create_pod", ToolEffect::Mutating);

    // One spend settles as failed (released); one is left in flight (crash).
    let released = guard.admit(&tool, Some(spend(300)), &no_args()).unwrap();
    guard
        .settle(released, ActionOutcome::Failed { detail: "x".into() })
        .unwrap();
    let _in_flight = guard.admit(&tool, Some(spend(450)), &no_args()).unwrap();
    drop(guard);

    let (_, records) = JsonlLog::open(&path).unwrap();
    let ledger = SpendLedger::replay(usd(1_000), &records);
    assert_eq!(ledger.remaining(), 550, "the unsettled 450 stays spent");
}

#[test]
fn a_truncated_final_line_is_skipped_and_the_next_append_starts_clean() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cos.jsonl");
    let reserved = CosLogRecord::SpendReserved(SpendReservation {
        reservation: "r-1".into(),
        tool: "mcp__runpod__create_pod".into(),
        amount: usd(200),
        reserved_at: NOW,
    });
    let full = serde_json::to_string(&reserved).unwrap();
    std::fs::write(&path, format!("{full}\n{}", &full[..full.len() / 2])).unwrap();

    let (log, records) = JsonlLog::open(&path).unwrap();
    assert_eq!(records, vec![reserved.clone()]);
    log.append(&reserved).unwrap();
    let (_, records) = JsonlLog::open(&path).unwrap();
    assert_eq!(records.len(), 2, "the half line was cut off, not merged");

    // A damaged line in the middle still stops the rebuild.
    let damaged = dir.path().join("damaged.jsonl");
    std::fs::write(&damaged, format!("{{broken\n{full}\n")).unwrap();
    assert!(matches!(
        JsonlLog::open(&damaged),
        Err(CosError::AnnouncementLog { line: 1, .. })
    ));
}

#[test]
fn a_panicking_tool_settles_its_ticket_and_keeps_the_spend_counted() {
    let log = Arc::new(MemoryLog::default());
    let guard = Arc::new(guard_on(shipped_profile(), 1_000, log.clone()));
    let (registry, _) = registry_with(
        guard.clone(),
        &[("mcp__runpod__create_pod", ToolEffect::Mutating, true)],
    );
    let input = ToolInput::new()
        .with_param("gpu", "H100")
        .with_value(DECLARATION_PARAM, serde_json::to_value(spend(250)).unwrap());
    let engine = engine();

    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        registry.execute("mcp__runpod__create_pod", input, &engine)
    }));
    assert!(unwound.is_err(), "the tool panicked");

    assert_eq!(guard.pending_admissions(), 0, "no leaked ticket");
    assert_eq!(
        guard.remaining_budget(),
        750,
        "an unknown outcome stays spent"
    );
    let records = log.records();
    let [
        CosLogRecord::SpendReserved(_),
        CosLogRecord::SpendSettled(SpendSettlement { spent: true, .. }),
        CosLogRecord::Announcement(announcement),
    ] = records.as_slice()
    else {
        panic!("expected reserve, settle(spent), announce; got {records:?}");
    };
    assert!(matches!(
        announcement.outcome,
        ActionOutcome::Unknown { .. }
    ));
    assert_eq!(SpendLedger::replay(usd(1_000), &records).remaining(), 750);
}

#[test]
fn concurrent_spends_never_overdraw_and_log_lines_never_interleave() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cos.jsonl");
    let (log, _) = JsonlLog::open(&path).unwrap();
    let guard = Arc::new(guard_on(shipped_profile(), 1_000, Arc::new(log)));
    let tool = manifest("mcp__runpod__create_pod", ToolEffect::Mutating);

    let admitted: usize = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..12)
            .map(|_| {
                let guard = guard.clone();
                let tool = tool.clone();
                scope.spawn(move || {
                    let mut input = ToolInput::new()
                        .with_value(DECLARATION_PARAM, serde_json::to_value(spend(150)).unwrap());
                    match ToolCallGuard::admit(guard.as_ref(), &tool, &mut input) {
                        Ok(ticket) => {
                            ToolCallGuard::settle(
                                guard.as_ref(),
                                ticket,
                                CallOutcome::Succeeded,
                                "pod up",
                            )
                            .unwrap();
                            1
                        }
                        Err(refusal) => {
                            assert!(refusal.contains("akh::cos::budget_exceeded"), "{refusal}");
                            0
                        }
                    }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });

    assert_eq!(admitted, 6, "1000 / 150 = 6 spends fit");
    assert_eq!(guard.remaining_budget(), 100);
    assert_eq!(guard.pending_admissions(), 0);
    // Every line parses: 6 reservations, 6 settlements, 6 announcements.
    let (_, records) = JsonlLog::open(&path).unwrap();
    assert_eq!(records.len(), 18);
    assert_eq!(SpendLedger::replay(usd(1_000), &records).remaining(), 100);
}

// ── Config: log placement, fail-closed fallback ─────────────────────────────

fn config_in(dir: &Path, profile: PathBuf, log: PathBuf) -> ChiefOfStaffConfig {
    ChiefOfStaffConfig {
        profile,
        repository: dir.to_path_buf(),
        announcements: log,
        budget: usd(1_000),
    }
}

#[test]
fn a_log_inside_the_write_sandbox_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let scratch = dir.path().join("scratch");
    std::fs::create_dir_all(&scratch).unwrap();
    let profile = skill_dir().join("permission-profile.json");

    let inside = config_in(dir.path(), profile.clone(), scratch.join("cos.jsonl"));
    assert!(matches!(
        CosGuard::from_config(&inside, std::slice::from_ref(&scratch)),
        Err(CosError::LogInsideSandbox { .. })
    ));
    let outside = config_in(dir.path(), profile, dir.path().join("state/cos.jsonl"));
    std::fs::create_dir_all(dir.path().join("state")).unwrap();
    assert!(CosGuard::from_config(&outside, &[scratch]).is_ok());
}

#[test]
fn a_guard_that_fails_to_load_stays_read_only_whatever_the_permission_mode() {
    let dir = tempfile::tempdir().unwrap();
    let config = config_in(
        dir.path(),
        dir.path().join("missing-profile.json"),
        dir.path().join("cos.jsonl"),
    );
    let install = GuardInstall::from_config(&config, &[]);
    assert!(matches!(
        install,
        GuardInstall::FailClosed {
            error: CosError::ProfileRead { .. },
            ..
        }
    ));
    let (registry, calls) = registry_with(
        install.guard(),
        &[
            ("memory_write", ToolEffect::Mutating, false),
            ("kg_query", ToolEffect::ReadOnly, false),
        ],
    );
    // An ACP client switching the mode back to bypass does not lift it.
    registry.set_permission_mode(PermissionMode::Bypass);
    let engine = engine();

    let write = registry
        .execute("memory_write", ToolInput::new(), &engine)
        .unwrap();
    assert!(!write.success);
    assert!(
        write.result.contains("akh::cos::guard_unavailable"),
        "{}",
        write.result
    );
    let read = registry
        .execute("kg_query", ToolInput::new(), &engine)
        .unwrap();
    assert!(read.success, "{}", read.result);
    assert_eq!(calls.lock().unwrap().len(), 1, "only the read ran");
}

// ── Catalog and repair ──────────────────────────────────────────────────────

#[test]
fn guarded_akhs_advertise_the_declaration_parameter() {
    let (guard, _) = guard();
    let (registry, _) = registry_with(
        guard,
        &[
            ("shell_exec", ToolEffect::Mutating, false),
            ("file_io", ToolEffect::Mutating, false),
            ("kg_query", ToolEffect::ReadOnly, false),
        ],
    );
    let manifests = registry.list_advertised_manifests();
    let param = |tool: &str| {
        manifests
            .iter()
            .find(|m| m.name == tool)
            .unwrap()
            .parameters
            .iter()
            .find(|p| p.name == DECLARATION_PARAM)
            .cloned()
    };
    let shell = param("shell_exec").expect("T3 tools advertise the declaration");
    assert!(shell.required);
    assert!(!param("file_io").unwrap().required, "overwrites only");
    assert!(param("kg_query").is_none());

    // The advertised fragment accepts what the guard accepts.
    let fragment = jsonschema::validator_for(shell.schema.as_ref().unwrap()).unwrap();
    assert_valid_against(&fragment, "advertised", &deploy_declaration(MAIN_SHA));
}

#[test]
fn argument_repair_keeps_the_declaration() {
    let catalog = vec![json!({
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "run",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    })];
    let declaration = deploy_declaration(MAIN_SHA);
    let repaired = crate::agent::organ::repair_known_call_args(
        "shell_exec",
        &json!({"comand": "ls", DECLARATION_PARAM: declaration.clone()}),
        &catalog,
    )
    .expect("the misspelt parameter is repaired");
    assert_eq!(repaired.arguments["command"], json!("ls"));
    assert_eq!(repaired.arguments[DECLARATION_PARAM], declaration);
}

// ── Gates and profile validation ────────────────────────────────────────────

#[test]
fn a_gated_tier_refuses_with_the_proposal_option() {
    let mut profile = shipped_profile();
    profile
        .grants
        .insert(AutonomyTier::OperatorClass, TierGrant::Gated);
    let (guard, _) = guard_with(profile, 1_000);
    let err = guard
        .admit(
            &manifest("shell_exec", ToolEffect::Mutating),
            declared(deploy_declaration(MAIN_SHA)),
            &no_args(),
        )
        .unwrap_err();
    assert!(matches!(err, CosError::TierGated { .. }));
    assert!(err.help().unwrap().to_string().contains("cos-proposal@1"));
}

#[test]
fn a_declaration_below_the_tool_tier_is_refused() {
    let (guard, _) = guard();
    let err = guard
        .admit(
            &manifest("shell_exec", ToolEffect::Mutating),
            Some(Declaration::Coordination {
                receipts: vec![receipt(ReceiptKind::WorkItem, "x")],
            }),
            &no_args(),
        )
        .unwrap_err();
    assert!(matches!(err, CosError::DeclarationBelowTier { .. }));
}

#[test]
fn a_profile_with_the_wrong_contract_is_rejected() {
    let text = json!({
        "contract": "lilith/cos-permission-profile@2",
        "akh": "lilith", "main_ref": "origin/main", "grants": {}, "tools": {}
    })
    .to_string();
    let err = CosPermissionProfile::from_json("inline", &text).unwrap_err();
    assert!(matches!(err, CosError::ProfileParse { .. }));
}

#[test]
fn commit_refs_are_found_structurally() {
    let refs: BTreeSet<String> = policy::commit_refs_in(&args(json!({
        "command": format!("crane copy x:main-{MAIN_SHA} y && echo 1234567 deadbee cafe"),
        "nested": [{"image": format!("akhomed-main-{BRANCH_SHA}")}],
        "id": "01a0e270-0288-7ac1-b297-a45375d4e6e6",
    })))
    .into_iter()
    .map(|sha| sha.to_string())
    .collect();
    // Digits-only and letters-only tokens are not SHAs; full UUIDs are ignored.
    assert_eq!(
        refs,
        BTreeSet::from([MAIN_SHA.to_string(), BRANCH_SHA.to_string()])
    );
}
