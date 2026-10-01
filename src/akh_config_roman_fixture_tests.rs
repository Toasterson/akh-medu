//! Lilith C2 (ADR 116): prove that the `lilith` workspace TOML rendered by
//! `deploy/roman-mini/configure_core.py` loads through the same entry point
//! akhomed uses for workspaces (`AkhConfig::load` = `load_checked` plus a
//! warning per unknown key), with every intended key present.
//!
//! Serde silently drops a key that lands in the wrong table (the akh TOML
//! table-ordering trap), so these tests also require `load_checked` to report
//! no unrecognized key at all.
//!
//! The input is `deploy/roman-mini/fixtures/golden/lilith.toml`. The Python
//! renderer test (`test_full_akh.RomanGoldenTests`) requires the renderer's
//! output to equal that fixture byte for byte, so a renderer change that drops
//! a key fails there or here. Set `AKH_LILITH_WORKSPACE_TOML=<path>` to check a
//! real render instead (its profile must be installed at the rendered path).
//!
//! `dojo_teacher_workspace_loads_sealed` does the same for the Lilith C4 dojo
//! teacher (`fixtures/golden/dojo-teacher.toml`, `DojoTeacherRenderTests`).

use std::path::{Path, PathBuf};

use super::*;
use crate::agent::chief_of_staff::GuardInstall;
use crate::agent::djehuty::BoardAuth;
use crate::agent::tools::mcp_config::McpServerAuth;

/// The rendered fixture; `{root}` stands for the deployment root.
const LILITH_FIXTURE: &str = include_str!("../deploy/roman-mini/fixtures/golden/lilith.toml");
const COS_PROFILE: &str = include_str!("../skills/chief-of-staff/permission-profile.json");
const ROMAN_FIXTURES: [(&str, &str); 3] = [
    (
        "janus",
        include_str!("../deploy/roman-mini/fixtures/golden/janus.toml"),
    ),
    (
        "minerva",
        include_str!("../deploy/roman-mini/fixtures/golden/minerva.toml"),
    ),
    (
        "vesta",
        include_str!("../deploy/roman-mini/fixtures/golden/vesta.toml"),
    ),
];
const PROFILE_REL: &str =
    "core-data/akh-medu/workspaces/lilith/kg/skills/chief-of-staff/permission-profile.json";

/// A deployment root holding the workspace TOML: either the fixture with
/// `{root}` bound to a temp dir (and the profile installed where it points),
/// or the operator-supplied real render.
struct Rendered {
    toml: PathBuf,
    root: PathBuf,
    _dir: Option<tempfile::TempDir>,
}

fn rendered_lilith() -> Rendered {
    if let Some(path) = std::env::var_os("AKH_LILITH_WORKSPACE_TOML") {
        let toml = PathBuf::from(path);
        // <root>/core-config/akh-medu/workspaces/lilith.toml
        let root = toml
            .ancestors()
            .nth(4)
            .expect(
                "AKH_LILITH_WORKSPACE_TOML is <root>/core-config/akh-medu/workspaces/lilith.toml",
            )
            .to_path_buf();
        return Rendered {
            toml,
            root,
            _dir: None,
        };
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let text = LILITH_FIXTURE.replace("{root}", &root.display().to_string());
    let toml = root.join("core-config/akh-medu/workspaces/lilith.toml");
    std::fs::create_dir_all(toml.parent().unwrap()).unwrap();
    std::fs::write(&toml, text).unwrap();
    let profile = root.join(PROFILE_REL);
    std::fs::create_dir_all(profile.parent().unwrap()).unwrap();
    std::fs::write(&profile, COS_PROFILE).unwrap();
    Rendered {
        toml,
        root,
        _dir: Some(dir),
    }
}

fn load_without_unknown_keys(path: &Path) -> AkhConfig {
    let (cfg, unknown) = AkhConfig::load_checked(path)
        .unwrap_or_else(|e| panic!("{} does not load: {e:?}", path.display()));
    assert!(
        unknown.is_empty(),
        "{}: serde dropped keys: {:?}",
        path.display(),
        unknown.iter().map(|k| k.to_string()).collect::<Vec<_>>()
    );
    cfg
}

#[test]
fn lilith_workspace_loads_with_every_intended_key() {
    let rendered = rendered_lilith();
    let cfg = load_without_unknown_keys(&rendered.toml);
    let root = &rendered.root;

    assert_eq!(cfg.name(), "lilith");
    assert_eq!(cfg.backend, ChatBacking::OpenAi);
    assert_eq!(cfg.model.as_deref(), Some("lilith"));
    let raw: toml::Value =
        toml::from_str(&std::fs::read_to_string(&rendered.toml).unwrap()).unwrap();
    assert!(
        raw.get("ollama_endpoint").is_none(),
        "no stale [ollama_endpoint]"
    );

    // [openai_endpoint], through the same resolution akhomed's turn uses.
    let endpoint = cfg
        .openai_endpoint
        .as_ref()
        .expect("[openai_endpoint] loaded");
    assert!(
        endpoint.base_url.starts_with("https://") && endpoint.base_url.ends_with("/v1"),
        "base_url: {}",
        endpoint.base_url
    );
    assert_eq!(endpoint.api_key.as_deref(), Some("env:LILITH_OPENAI_KEY"));
    assert_eq!(endpoint.context_window, Some(49152));
    let openai = cfg
        .to_openai_config()
        .expect("openai config resolves")
        .expect("backend is openai");
    assert_eq!(openai.model, "lilith");
    assert_eq!(openai.base_url, endpoint.base_url);
    assert_eq!(openai.context_window, Some(49152));

    // [chief_of_staff] (deny_unknown_fields: profile, repository,
    // announcements, budget, gated_tiers).
    let cos = cfg
        .chief_of_staff
        .as_ref()
        .expect("[chief_of_staff] loaded");
    assert_eq!(cos.profile, root.join(PROFILE_REL));
    assert_eq!(cos.repository, root.join("lilith/ancestry/akh-medu.git"));
    assert_eq!(
        cos.announcements,
        root.join("state/lilith/cos-announcements.jsonl")
    );
    assert_eq!(cos.budget.currency.as_str(), "USD");
    if std::env::var_os("AKH_LILITH_WORKSPACE_TOML").is_none() {
        assert_eq!(cos.budget.minor_units, 6673, "the fixture's budget input");
    }

    // Operator decision 2026-10-01: T3 is gated (read-only) until C2 round 2,
    // and the ACP intake strips the T3 client's injected preamble.
    assert_eq!(
        cos.gated_tiers,
        [crate::agent::chief_of_staff::AutonomyTier::OperatorClass]
    );
    assert!(cfg.acp.strip_client_preamble, "[acp] strip_client_preamble");

    // The guard akhomed installs from this table: the log is outside her
    // write root (the engine's <workspace>/kg/scratch) and the installed
    // profile loads, so she does not fall back to read-only.
    let write_roots = vec![root.join("core-data/akh-medu/workspaces/lilith/kg/scratch")];
    std::fs::create_dir_all(cos.announcements.parent().unwrap()).unwrap();
    match GuardInstall::from_config(cos, &write_roots) {
        GuardInstall::Active(guard) => {
            assert_eq!(guard.remaining_budget(), cos.budget.minor_units);
            use crate::agent::chief_of_staff::{AutonomyTier, TierGrant};
            assert_eq!(
                guard.profile().grant(AutonomyTier::OperatorClass),
                TierGrant::Gated
            );
            assert_eq!(
                guard.profile().grant(AutonomyTier::Coordination),
                TierGrant::Autonomous
            );
        }
        GuardInstall::FailClosed { error, .. } => panic!("CosGuard failed closed: {error:?}"),
    }

    // [board]: her own Board identity.
    let board = cfg.board.as_ref().expect("[board] loaded");
    assert_eq!(board.url, "http://127.0.0.1:18089");
    assert_eq!(board.auth, BoardAuth::TokenEnv("ROMAN_BOARD_LILITH".into()));

    // [[mcp.servers]]: anima as the anima-lilith identity (her PAT), djehuty
    // as board-lilith, seshat.
    let servers: std::collections::BTreeMap<&str, _> = cfg
        .mcp
        .servers
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();
    assert_eq!(
        servers.keys().copied().collect::<Vec<_>>(),
        ["anima", "djehuty", "seshat"]
    );
    assert_eq!(
        servers["anima"].auth,
        McpServerAuth::TokenEnv("LILITH_ANIMA_TOKEN".into())
    );
    assert_eq!(servers["anima"].url, "https://anima.wegmueller.it/mcp");
    assert_eq!(
        servers["djehuty"].auth,
        McpServerAuth::TokenEnv("ROMAN_BOARD_LILITH".into())
    );
    assert_eq!(servers["djehuty"].url, "http://127.0.0.1:18089/mcp");
    assert_eq!(
        servers["seshat"].auth,
        McpServerAuth::TokenEnv("ROMAN_ARCHIVE_TOKEN".into())
    );
    assert!(servers["seshat"].allow_insecure_http);
    assert!(
        servers.values().all(|s| s.tools.is_none()),
        "no allowlist: her credentials and the CosGuard bound her"
    );

    assert!(cfg.autonomy.enabled);
    assert!(
        cfg.system_prompt
            .as_deref()
            .is_some_and(|p| p.contains("Chief of Staff") && !p.contains("{mcp_servers}")),
        "system_prompt carries her role and the skill pack"
    );
}

#[test]
fn roman_workspaces_load_without_dropped_keys() {
    let dir = tempfile::tempdir().expect("tempdir");
    for (role, fixture) in ROMAN_FIXTURES {
        let path = dir.path().join(format!("{role}.toml"));
        std::fs::write(
            &path,
            fixture.replace("{root}", &dir.path().display().to_string()),
        )
        .unwrap();
        let cfg = load_without_unknown_keys(&path);
        assert_eq!(cfg.name(), role);
        assert_eq!(cfg.backend, ChatBacking::Ollama);
        assert!(cfg.chief_of_staff.is_none());
        assert!(!cfg.acp.strip_client_preamble, "{role}: no preamble strip");
        // ADR 123: no per-akh budget is rendered; the GLM-5.3 catalog row sets it.
        assert!(
            cfg.autonomy.budget.is_empty(),
            "{role}: no [autonomy.budget] override"
        );
        let profile = cfg
            .resolve_model_profile()
            .expect("the Roman profile resolves");
        assert_eq!(profile.catalog_entry.as_deref(), Some("glm-5.3"));
        assert_eq!(
            profile.daily_budget(),
            crate::agent::autonomy::AutonomyBudget {
                max_tokens_per_day: 1_500_000,
                max_turns_per_day: 96,
            }
        );
        assert_eq!(profile.max_tokens_per_wake.map(|s| s.value), Some(250_000));
        assert_eq!(profile.max_tools_per_wake.map(|s| s.value), Some(24));
    }
}

/// ADR 123: Lilith's rendered workspace resolves the local Gemma 4 row, with
/// her `[openai_endpoint] context_window` as the per-akh window.
#[test]
fn lilith_workspace_resolves_the_local_gemma_profile() {
    let rendered = rendered_lilith();
    let cfg = load_without_unknown_keys(&rendered.toml);
    let profile = cfg.resolve_model_profile().expect("resolves");
    assert_eq!(profile.catalog_entry.as_deref(), Some("gemma-4-12b"));
    assert_eq!(profile.max_tools_per_wake.map(|s| s.value), Some(16));
    // ADR 123 amendment: her interactive turns offer 16 tools, all of them the
    // Chief-of-Staff pinned core, so the tool block never changes.
    assert_eq!(profile.max_tools_per_turn.map(|s| s.value), Some(16));
    assert_eq!(cfg.pinned_tools.len(), 16, "{:?}", cfg.pinned_tools);
    assert!(cfg.pinned_tools.is_sorted(), "{:?}", cfg.pinned_tools);
    assert!(
        cfg.pinned_tools
            .iter()
            .any(|t| t == "mcp__anima__get_work_item_detail")
    );
    assert!(
        cfg.pinned_tools
            .iter()
            .any(|t| t == "mcp__anima__list_projects")
    );
    assert!(
        !cfg.pinned_tools
            .iter()
            .any(|t| t == "mcp__anima__get_work_item")
    );
    let openai = cfg.to_openai_config().unwrap().unwrap();
    assert_eq!(openai.context_window, Some(49_152));
    assert_eq!(openai.timeout_secs, 600);
}

/// Lilith C4 (ADR 116): the sealed dojo teacher `configure_core.py` renders
/// behind `config/dojo-teacher.json`. It must load through `load_checked`,
/// which runs `validate_tool_surface`, with no key dropped.
const DOJO_TEACHER_FIXTURE: &str =
    include_str!("../deploy/roman-mini/fixtures/golden/dojo-teacher.toml");

#[test]
fn dojo_teacher_workspace_loads_sealed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("dojo-teacher.toml");
    std::fs::write(&path, DOJO_TEACHER_FIXTURE).unwrap();
    let cfg = load_without_unknown_keys(&path);
    assert_eq!(cfg.name(), "dojo-teacher");
    assert_eq!(cfg.tool_surface, ToolSurface::SessionOnly);
    assert_eq!(cfg.backend, ChatBacking::Ollama);
    assert_eq!(cfg.context_strategy.as_deref(), Some("passthrough"));
    assert!(!cfg.capture_transcripts && !cfg.autonomy.enabled);
    assert!(matches!(cfg.exec, Some(ExecBackendConfig::None)));
    assert!(cfg.mcp.servers.is_empty() && cfg.mcp.require_pushed_mcp);
    assert!(
        cfg.board.is_none() && cfg.chief_of_staff.is_none() && cfg.related_workspaces.is_empty()
    );
    assert!(
        !cfg.acp.strip_client_preamble,
        "the dojo teacher keeps its raw prompts and has no CoS gate"
    );
    // ADR 123: the teacher's turns run on the GLM-5.3 row; autonomy stays off.
    let profile = cfg.resolve_model_profile().expect("resolves");
    assert_eq!(profile.catalog_entry.as_deref(), Some("glm-5.3"));
    assert_eq!(profile.turn_deadline_secs.value, 1_800);
}
