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
    assert_eq!(endpoint.context_window, Some(32768));
    let openai = cfg
        .to_openai_config()
        .expect("openai config resolves")
        .expect("backend is openai");
    assert_eq!(openai.model, "lilith");
    assert_eq!(openai.base_url, endpoint.base_url);
    assert_eq!(openai.context_window, Some(32768));

    // [chief_of_staff] (deny_unknown_fields: profile, repository,
    // announcements, budget).
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

    // The guard akhomed installs from this table: the log is outside her
    // write root (the engine's <workspace>/kg/scratch) and the installed
    // profile loads, so she does not fall back to read-only.
    let write_roots = vec![root.join("core-data/akh-medu/workspaces/lilith/kg/scratch")];
    std::fs::create_dir_all(cos.announcements.parent().unwrap()).unwrap();
    match GuardInstall::from_config(cos, &write_roots) {
        GuardInstall::Active(guard) => {
            assert_eq!(guard.remaining_budget(), cos.budget.minor_units);
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
    }
}
