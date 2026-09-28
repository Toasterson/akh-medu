//! Lilith C1 exit criteria (ADR 116, plan slice C1): schema tests, a T3
//! action taken and announced with receipts + rollback path, a non-main
//! deploy refused by the rule check, a T2 post carrying receipts, adapter
//! activation refused.

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use miette::Diagnostic;
use serde_json::{Value, json};

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolRegistry, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolSource,
};
use crate::engine::{Engine, EngineConfig};
use crate::vsa::Dimension;

use super::contracts::*;
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

fn assert_valid(name: &str, instance: &Value) {
    let validator = validator(name);
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .map(|e| e.to_string())
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:?}\n{instance:#}");
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

fn shipped_profile() -> CosPermissionProfile {
    CosPermissionProfile::load(&skill_dir().join("permission-profile.json")).unwrap()
}

/// Ancestry oracle over a fixed set of main commits.
struct FakeMain(BTreeSet<String>);

impl AncestryOracle for FakeMain {
    fn is_ancestor(&self, sha: &CommitSha, _main_ref: &str) -> CosResult<bool> {
        Ok(self.0.contains(sha.as_str()))
    }
}

fn guard_with(
    profile: CosPermissionProfile,
    budget: u64,
) -> (Arc<CosGuard>, Arc<MemoryAnnouncementSink>) {
    let sink = Arc::new(MemoryAnnouncementSink::default());
    let guard = CosGuard::new(
        profile,
        SpendLedger::new(usd(budget)),
        Box::new(FakeMain(BTreeSet::from([MAIN_SHA.to_string()]))),
        sink.clone(),
    )
    .with_clock(fixed_clock);
    (Arc::new(guard), sink)
}

fn guard() -> (Arc<CosGuard>, Arc<MemoryAnnouncementSink>) {
    guard_with(shipped_profile(), 5_000)
}

/// A tool that records the inputs it actually received.
struct RecordingTool {
    name: &'static str,
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
        Ok(ToolOutput::ok("done"))
    }

    fn manifest(&self) -> ToolManifest {
        manifest(self.name, ToolEffect::Mutating)
    }
}

fn registry_with(
    guard: Arc<CosGuard>,
    names: &[&'static str],
) -> (ToolRegistry, Arc<Mutex<Vec<ToolInput>>>) {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let registry = ToolRegistry::new();
    for name in names {
        registry.register(Box::new(RecordingTool {
            name,
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

fn deploy_declaration(artifact_sha: &str) -> Value {
    json!({
        "tier": "operator-class",
        "action": {"kind": "deploy", "artifact_sha": artifact_sha, "target": "akhomed@archibald"},
        "receipts": [{"kind": "run-record", "reference": "ci/job/8812", "observed_at": NOW}],
        "reason": "CI green on the fix; the running binary predates it",
        "rollback": {"kind": "revert", "steps": "redeploy akhomed-main-1a2b3c4 and restart"}
    })
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
            action: deploy.clone(),
            receipts: receipts(ReceiptKind::AncestryCheck, "git merge-base --is-ancestor"),
            reason: "fix landed".into(),
            rollback: rollback.clone(),
            rule_checks: vec![RuleCheck {
                rule: StandingRule::MainOnlyDeploy,
                evidence: "exit 0".into(),
            }],
            outcome: ActionOutcome::Succeeded {
                detail: "done".into(),
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
            action: OperatorAction::ActivateAdapter {
                adapter: "lilith-cos-r1".into(),
            },
            receipts: receipts(ReceiptKind::RunRecord, "runs/c7/report"),
            reason: "sealed eval beats the parent on every family".into(),
            rollback: RollbackPath::Revert {
                steps: "switch the served adapter back to the parent".into(),
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
}

#[test]
fn schemas_and_types_reject_missing_standing_fields() {
    let documents = sample_documents();
    let wire = |index: usize| serde_json::to_value(&documents[index]).unwrap();

    // An action item without its cost of skipping.
    let mut action_item = wire(1);
    action_item
        .as_object_mut()
        .unwrap()
        .remove("cost_of_skipping");
    // A briefing item without receipts (Honesty).
    let mut briefing = wire(0);
    briefing["blocked"][0]["receipts"] = json!([]);
    // An announcement without a rollback path.
    let mut announcement = wire(2);
    announcement.as_object_mut().unwrap().remove("rollback");
    // A deploy naming something that is not a SHA.
    let mut not_a_sha = wire(2);
    not_a_sha["action"]["artifact_sha"] = json!("--upload-pack=evil");
    // Guidance without a citation.
    let mut uncited = wire(4);
    uncited["citations"] = json!([]);
    // A language that is not a BCP 47 tag.
    let mut language = wire(5);
    language["language"] = json!("German");

    for (name, instance) in [
        ("cos-action-item", action_item),
        ("cos-briefing", briefing),
        ("cos-announcement", announcement),
        ("cos-announcement", not_a_sha),
        ("cos-guidance", uncited),
        ("cos-proposal", language),
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

// ── Exit criterion: T3 taken and announced ──────────────────────────────────

#[test]
fn t3_action_is_taken_and_announced_with_receipts_and_rollback_path() {
    let (guard, sink) = guard();
    let (registry, calls) = registry_with(guard, &["shell_exec"]);
    let input = ToolInput::new()
        .with_param("command", "deploy/akhomed.sh")
        .with_value(DECLARATION_PARAM, deploy_declaration(MAIN_SHA));

    let out = registry.execute("shell_exec", input, &engine()).unwrap();
    assert!(out.success, "{}", out.result);

    // The action ran, and the tool never saw the declaration.
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].get_value(DECLARATION_PARAM).is_none());
    assert_eq!(calls[0].get("command"), Some("deploy/akhomed.sh"));

    // Announced: her receipt plus the quoted ancestry check, reason, rollback.
    let announced = sink.announcements();
    assert_eq!(announced.len(), 1);
    let announcement = &announced[0];
    assert_eq!(announcement.akh, "lilith");
    assert_eq!(announcement.tool, "shell_exec");
    assert!(announcement.outcome.succeeded());
    assert!(!announcement.reason.is_empty());
    assert!(matches!(announcement.rollback, RollbackPath::Revert { .. }));
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
    let ancestry = &announcement.receipts.as_slice()[1].reference;
    assert_eq!(
        ancestry,
        &format!("git merge-base --is-ancestor {MAIN_SHA} origin/main")
    );
    let rules: Vec<StandingRule> = announcement.rule_checks.iter().map(|c| c.rule).collect();
    assert_eq!(
        rules,
        vec![StandingRule::ReceiptsPresent, StandingRule::MainOnlyDeploy]
    );

    // The announcement is a valid `lilith/cos-announcement@1` document.
    let wire = serde_json::to_value(CosDocument::Announcement(announcement.clone())).unwrap();
    assert_valid("cos-announcement", &wire);
}

#[test]
fn announcements_persist_to_jsonl_and_rebuild_the_spend_ledger() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("cos-announcements.jsonl");
    let sink = Arc::new(JsonlAnnouncementSink::new(&log));
    let guard = CosGuard::new(
        shipped_profile(),
        SpendLedger::new(usd(1_000)),
        Box::new(FakeMain(BTreeSet::new())),
        sink.clone(),
    )
    .with_clock(fixed_clock);
    let spend = Declaration::OperatorClass {
        action: OperatorAction::Spend {
            amount: usd(700),
            purpose: "akh-medu-lilith-serve, 2 h".into(),
        },
        receipts: vec![receipt(ReceiptKind::RunRecord, "runpod/pricing")],
        reason: "C2 operator session".into(),
        rollback: RollbackPath::Irreversible {
            mitigation: "pod auto-terminates after 2 h".into(),
        },
    };
    let admission = guard
        .admit(
            &manifest("mcp__runpod__create_pod", ToolEffect::Mutating),
            Some(spend),
        )
        .unwrap();
    guard
        .settle(
            admission,
            ActionOutcome::Succeeded {
                detail: "pod up".into(),
            },
        )
        .unwrap();

    let line = std::fs::read_to_string(&log).unwrap();
    assert_valid(
        "cos-announcement",
        &serde_json::from_str::<Value>(line.trim()).unwrap(),
    );
    let replayed = SpendLedger::replay(usd(1_000), &sink.read_all().unwrap());
    assert_eq!(replayed.remaining(), 300);
}

// ── Exit criterion: non-main deploy refused ─────────────────────────────────

#[test]
fn non_main_deploy_is_refused_by_the_rule_check_with_a_diagnostic() {
    let (guard, sink) = guard();
    let declaration: Declaration = serde_json::from_value(deploy_declaration(BRANCH_SHA)).unwrap();

    let err = guard
        .admit(
            &manifest("shell_exec", ToolEffect::Mutating),
            Some(declaration),
        )
        .unwrap_err();
    assert!(matches!(err, CosError::NonMainArtifact { .. }), "{err:?}");
    assert_eq!(err.standing_rule(), Some(StandingRule::MainOnlyDeploy));
    assert_eq!(
        err.code().unwrap().to_string(),
        "akh::cos::non_main_artifact"
    );
    let help = err.help().unwrap().to_string();
    assert!(help.contains("origin/main"), "{help}");

    // Through the registry: the tool does not run, nothing is announced, and
    // the refusal carries the diagnostic code and the options.
    let (registry, calls) = registry_with(guard, &["shell_exec"]);
    let input = ToolInput::new().with_value(DECLARATION_PARAM, deploy_declaration(BRANCH_SHA));
    let out = registry.execute("shell_exec", input, &engine()).unwrap();
    assert!(!out.success);
    assert!(
        out.result.contains("akh::cos::non_main_artifact"),
        "{}",
        out.result
    );
    assert!(calls.lock().unwrap().is_empty());
    assert!(sink.announcements().is_empty());
}

#[test]
fn git_ancestry_decides_against_a_real_repository() {
    let git = |dir: &std::path::Path, args: &[&str]| {
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
        eprintln!("git not installed; GitAncestry exercised through FakeMain only");
        return;
    };
    assert!(init.status.success());
    let head = |dir: &std::path::Path| {
        let out = git(dir, &["rev-parse", "HEAD"]).unwrap();
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    };
    assert!(
        git(dir.path(), &["commit", "-q", "--allow-empty", "-m", "a"])
            .unwrap()
            .status
            .success()
    );
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
    assert!(
        git(dir.path(), &["commit", "-q", "--allow-empty", "-m", "b"])
            .unwrap()
            .status
            .success()
    );
    let on_branch = head(dir.path());

    let oracle = GitAncestry::new(dir.path());
    assert!(oracle.is_ancestor(&sha(&on_main), "origin/main").unwrap());
    assert!(!oracle.is_ancestor(&sha(&on_branch), "origin/main").unwrap());
    let unknown = oracle.is_ancestor(&sha("0123456789abcdef0123"), "origin/main");
    assert!(matches!(unknown, Err(CosError::AncestryUnknown { .. })));
}

// ── Exit criterion: T2 post carries receipts ────────────────────────────────

#[test]
fn t2_post_carries_receipts() {
    let (guard, sink) = guard();
    let post = manifest("mcp__anima__post_message", ToolEffect::Mutating);

    let admission = guard
        .admit(
            &post,
            Some(Declaration::Coordination {
                receipts: vec![receipt(ReceiptKind::WorkItem, "01a0e270-0288")],
            }),
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
    // T2 is autonomous: nothing waits for approval, nothing is announced.
    assert!(sink.announcements().is_empty());

    let no_receipts = guard
        .admit(&post, Some(Declaration::Coordination { receipts: vec![] }))
        .unwrap_err();
    assert!(matches!(no_receipts, CosError::MissingReceipts { .. }));
    assert_eq!(
        no_receipts.standing_rule(),
        Some(StandingRule::ReceiptsPresent)
    );
    let undeclared = guard.admit(&post, None).unwrap_err();
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
    ] {
        let admission = guard
            .admit(&manifest(name, ToolEffect::Mutating), None)
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

// ── Exit criterion: adapter activation refused ──────────────────────────────

#[test]
fn adapter_activation_is_refused() {
    let (guard, sink) = guard();

    // By tool class.
    let by_tool = guard
        .admit(
            &manifest("lilith_adapter_activate", ToolEffect::Mutating),
            None,
        )
        .unwrap_err();
    assert!(matches!(by_tool, CosError::OutsideGrant { .. }));

    // By declared action, even on a tool she may otherwise use for T3.
    let by_action = Declaration::OperatorClass {
        action: OperatorAction::ActivateAdapter {
            adapter: "lilith-cos-r1".into(),
        },
        receipts: vec![receipt(ReceiptKind::RunRecord, "runs/c7/report")],
        reason: "sealed eval passed".into(),
        rollback: RollbackPath::Revert {
            steps: "serve the parent".into(),
        },
    };
    let (registry, calls) = registry_with(guard.clone(), &["shell_exec"]);
    let input =
        ToolInput::new().with_value(DECLARATION_PARAM, serde_json::to_value(&by_action).unwrap());
    let out = registry.execute("shell_exec", input, &engine()).unwrap();
    assert!(!out.success);
    assert!(
        out.result.contains("akh::cos::outside_grant"),
        "{}",
        out.result
    );
    assert!(calls.lock().unwrap().is_empty());
    assert!(sink.announcements().is_empty());
}

// ── Other standing rules ────────────────────────────────────────────────────

#[test]
fn spend_beyond_budget_is_refused_and_a_failed_spend_is_released() {
    let (guard, _) = guard_with(shipped_profile(), 1_000);
    let tool = manifest("mcp__runpod__create_pod", ToolEffect::Mutating);
    let spend = |minor_units| Declaration::OperatorClass {
        action: OperatorAction::Spend {
            amount: usd(minor_units),
            purpose: "H100 hour".into(),
        },
        receipts: vec![receipt(ReceiptKind::RunRecord, "runpod/pricing")],
        reason: "C6 smoke".into(),
        rollback: RollbackPath::Irreversible {
            mitigation: "auto-terminate".into(),
        },
    };

    let too_much = guard.admit(&tool, Some(spend(1_001))).unwrap_err();
    assert!(matches!(
        too_much,
        CosError::BudgetExceeded {
            remaining: 1_000,
            ..
        }
    ));
    assert_eq!(too_much.standing_rule(), Some(StandingRule::WithinBudget));

    let admission = guard.admit(&tool, Some(spend(600))).unwrap();
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
}

#[test]
fn irreversible_deletion_needs_an_inspected_target_and_a_verified_copy() {
    let (guard, _) = guard();
    let tool = manifest("shell_exec", ToolEffect::Mutating);
    let target = TargetVerification {
        target: "runpod-volume:4wgri7h01a".into(),
        digest: "sha256:aa".into(),
        receipt: receipt(ReceiptKind::Digest, "4wgri7h01a/SHA256SUMS"),
    };
    let delete = |copy: Option<CopyVerification>| Declaration::OperatorClass {
        action: OperatorAction::Delete {
            target: target.clone(),
            reversibility: Reversibility::Irreversible,
            copy,
        },
        receipts: vec![receipt(ReceiptKind::Document, "operator decision")],
        reason: "volume retired".into(),
        rollback: RollbackPath::RestoreFromCopy {
            location: "nfs:/export/toast/akh-medu-runs/babyllm".into(),
        },
    };
    let copy = |location: &str, digest: &str| CopyVerification {
        location: location.into(),
        digest: digest.into(),
        receipt: receipt(ReceiptKind::Digest, location),
    };
    let gap = |declaration| match guard.admit(&tool, Some(declaration)) {
        Err(CosError::UnverifiedDeletion { gap, .. }) => gap,
        other => panic!("expected a verified-deletion refusal, got {other:?}"),
    };

    assert_eq!(gap(delete(None)), DeletionGap::NoVerifiedCopy);
    assert_eq!(
        gap(delete(Some(copy("nfs:/babyllm", "sha256:bb")))),
        DeletionGap::CopyDigestMismatch
    );
    assert_eq!(
        gap(delete(Some(copy("runpod-volume:4wgri7h01a", "sha256:aa")))),
        DeletionGap::CopyIsTarget
    );
    let admitted = guard
        .admit(&tool, Some(delete(Some(copy("nfs:/babyllm", "sha256:aa")))))
        .unwrap();
    let Admission::OperatorClass(authorized) = admitted else {
        panic!("expected an operator-class admission");
    };
    assert!(
        authorized
            .rule_checks
            .iter()
            .any(|c| c.rule == StandingRule::VerifiedDeletion)
    );
}

#[test]
fn a_gated_tier_refuses_with_the_proposal_option() {
    let mut profile = shipped_profile();
    profile
        .grants
        .insert(AutonomyTier::OperatorClass, TierGrant::Gated);
    let (guard, _) = guard_with(profile, 1_000);
    let declaration: Declaration = serde_json::from_value(deploy_declaration(MAIN_SHA)).unwrap();
    let err = guard
        .admit(
            &manifest("shell_exec", ToolEffect::Mutating),
            Some(declaration),
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
