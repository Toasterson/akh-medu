//! OpenClaw workspace discovery tool (OpenClaw-specific; switch-gated).
//!
//! Resolves the akh's *own* OpenClaw workspace — the agent in
//! `~/.openclaw/openclaw.json` whose `id` matches this akh's id — and returns a
//! typed inventory the LLM uses to plan a self-import: which persona files exist,
//! how many memory journals and skills are present, and whether the SQLite /
//! Lance stores are there. Read-only, with a secrets denylist so credential
//! material is never surfaced. Registered only when `openclaw_import` is on.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::fs::{Filesystem, LocalFs};

/// Persona / context markdown files an OpenClaw workspace may carry.
const PERSONA_FILES: &[&str] = &[
    "IDENTITY.md",
    "SOUL.md",
    "USER.md",
    "AGENTS.md",
    "MEMORY.md",
    "HEARTBEAT.md",
    "TOOLS.md",
];

/// Discover the akh's own OpenClaw workspace and inventory it.
pub struct OpenClawDiscoverTool {
    agent_id: String,
}

impl OpenClawDiscoverTool {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
        }
    }
}

/// Whether a path component looks like secret material that must never be surfaced.
fn is_secret_path(name: &str) -> bool {
    let lower = name.to_lowercase();
    const MARKERS: &[&str] = &[
        ".ssh",
        ".gnupg",
        "credentials",
        "secret",
        "secrets",
        ".pem",
        "id_rsa",
        ".gitconfig",
        "service-env",
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

/// Expand a leading `~` to `$HOME`.
fn expand_home(raw: &str) -> PathBuf {
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if raw == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(raw)
}

impl Tool for OpenClawDiscoverTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "openclaw_discover".into(),
            description: "Locate your own OpenClaw workspace and report an inventory of what can \
                          be imported (persona files, memory journals, skills, SQLite/Lance \
                          stores). Call this first."
                .into(),
            parameters: vec![ToolParam {
                name: "openclaw_home".into(),
                description: "OpenClaw root directory (default: ~/.openclaw).".into(),
                required: false,
            }],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let home = input
            .get("openclaw_home")
            .map(expand_home)
            .unwrap_or_else(|| expand_home("~/.openclaw"));

        // Read-only filesystem scoped to the OpenClaw root.
        let fs = LocalFs::read_only(vec![home.clone()]);

        let config_path = home.join("openclaw.json");
        let config_text = match fs.read_to_string(&config_path) {
            Ok(t) => t,
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "could not read {}: {e}",
                    config_path.display()
                )));
            }
        };
        let config: serde_json::Value = match serde_json::from_str(&config_text) {
            Ok(v) => v,
            Err(e) => {
                return Ok(ToolOutput::err(format!("openclaw.json is not valid JSON: {e}")));
            }
        };

        // Find the agent entry whose id == this akh's id.
        let agents = config
            .get("agents")
            .and_then(|a| a.get("list"))
            .and_then(|l| l.as_array());
        let Some(agents) = agents else {
            return Ok(ToolOutput::err("openclaw.json has no agents.list array"));
        };
        let entry = agents
            .iter()
            .find(|a| a.get("id").and_then(|i| i.as_str()) == Some(self.agent_id.as_str()));
        let Some(entry) = entry else {
            return Ok(ToolOutput::ok(
                serde_json::json!({
                    "found": false,
                    "agent_id": self.agent_id,
                    "message": format!(
                        "no OpenClaw agent with id \"{}\" found in {}",
                        self.agent_id,
                        config_path.display()
                    ),
                })
                .to_string(),
            ));
        };

        let ws_raw = entry
            .get("workspace")
            .and_then(|w| w.as_str())
            .unwrap_or("~/.openclaw/workspace");
        let workspace = expand_home(ws_raw);

        let inventory = build_inventory(&fs, &home, &workspace, &self.agent_id);
        Ok(ToolOutput::ok(inventory.to_string()))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "openclaw_discover".into(),
            description: "Locate your own OpenClaw workspace and inventory what can be imported."
                .into(),
            parameters: vec![ToolParamSchema::optional(
                "openclaw_home",
                "OpenClaw root directory (default: ~/.openclaw).",
            )],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadFilesystem]),
                description: "Reads the OpenClaw config + workspace directory (read-only).".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// Count markdown files directly under `dir` (non-recursive), skipping secrets.
fn count_md(fs: &LocalFs, dir: &Path) -> usize {
    fs.read_dir(dir)
        .map(|entries| {
            entries
                .iter()
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| !is_secret_path(n) && n.to_lowercase().ends_with(".md"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

/// Count `skills/*/SKILL.md` entries (one per skill subdir), skipping secrets.
fn count_skills(fs: &LocalFs, skills_dir: &Path) -> usize {
    fs.read_dir(skills_dir)
        .map(|entries| {
            entries
                .iter()
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| !is_secret_path(n))
                        .unwrap_or(false)
                        && fs.exists(&p.join("SKILL.md"))
                })
                .count()
        })
        .unwrap_or(0)
}

/// Build the typed inventory JSON.
fn build_inventory(
    fs: &LocalFs,
    home: &Path,
    workspace: &Path,
    agent_id: &str,
) -> serde_json::Value {
    let persona: Vec<&str> = PERSONA_FILES
        .iter()
        .copied()
        .filter(|f| fs.exists(&workspace.join(f)))
        .collect();

    let memory_dir = workspace.join("memory");
    let memory_count = count_md(fs, &memory_dir);

    let skills_dir = workspace.join("skills");
    let skills_count = count_skills(fs, &skills_dir);

    let sqlite_path = home.join("memory").join(format!("{agent_id}.sqlite"));
    let sqlite_present = fs.exists(&sqlite_path);

    let lance_dir = home.join("knowledge").join("agents").join(agent_id);
    let lance_present = fs.exists(&lance_dir);

    serde_json::json!({
        "found": true,
        "agent_id": agent_id,
        "workspace": workspace.display().to_string(),
        "persona_files": persona,
        "memory_journals": {
            "dir": memory_dir.display().to_string(),
            "count": memory_count,
            "glob": format!("{}/*.md", memory_dir.display()),
        },
        "skills": {
            "dir": skills_dir.display().to_string(),
            "count": skills_count,
            "glob": format!("{}/*/SKILL.md", skills_dir.display()),
        },
        "sqlite_memory": {
            "present": sqlite_present,
            "path": sqlite_path.display().to_string(),
            "hint": "read read-only via shell_exec: sqlite3 -readonly -json <path> '.schema'",
        },
        "lance_library": {
            "present": lance_present,
            "path": lance_dir.display().to_string(),
            "hint": "export to JSONL out-of-band via shell_exec, then read with file_io",
        },
        "next": "Read persona files with file_io and call identity_awaken; \
                 read each memory/*.md and call memory_write (origin=openclaw); \
                 read each SKILL.md and call skill_author.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    fn fixture() -> tempfile::TempDir {
        let tmp = tempfile::TempDir::new().unwrap();
        let home = tmp.path();
        let ws = home.join("workspace");
        // openclaw.json mapping id=tecton → the workspace dir (absolute).
        write(
            &home.join("openclaw.json"),
            &serde_json::json!({
                "agents": {"list": [
                    {"id": "tecton", "default": true, "workspace": ws.display().to_string()}
                ]}
            })
            .to_string(),
        );
        // Persona + memory + skill + a planted secret + sqlite.
        write(&ws.join("IDENTITY.md"), "# Tecton");
        write(&ws.join("memory/2026-05-08.md"), "a day");
        write(&ws.join("memory/credentials.md"), "SECRET token");
        write(&ws.join("skills/cluster-inspect/SKILL.md"), "---\nname: x\n---\nbody");
        write(&home.join("memory/tecton.sqlite"), "sqlitebytes");
        tmp
    }

    #[test]
    fn discovers_own_workspace_and_inventories() {
        let tmp = fixture();
        let tool = OpenClawDiscoverTool::new("tecton");
        let input = ToolInput::new().with_value("openclaw_home", tmp.path().display().to_string());
        let out = tool.execute(&test_engine(), input).unwrap();
        assert!(out.success, "{}", out.result);

        let v: serde_json::Value = serde_json::from_str(&out.result).unwrap();
        assert_eq!(v["found"], true);
        assert_eq!(v["memory_journals"]["count"], 1); // credentials.md excluded
        assert_eq!(v["skills"]["count"], 1);
        assert_eq!(v["sqlite_memory"]["present"], true);
        let persona = v["persona_files"].as_array().unwrap();
        assert!(persona.iter().any(|p| p == "IDENTITY.md"));
    }

    #[test]
    fn unknown_agent_reports_not_found() {
        let tmp = fixture();
        let tool = OpenClawDiscoverTool::new("nonexistent");
        let input = ToolInput::new().with_value("openclaw_home", tmp.path().display().to_string());
        let out = tool.execute(&test_engine(), input).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out.result).unwrap();
        assert_eq!(v["found"], false);
    }

    #[test]
    fn secret_marker_detection() {
        assert!(is_secret_path("credentials.md"));
        assert!(is_secret_path(".ssh"));
        assert!(is_secret_path("id_rsa"));
        assert!(!is_secret_path("2026-05-08.md"));
    }
}
