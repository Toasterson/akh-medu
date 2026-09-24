//! Skill authoring tool: register a skill from a name + description + body.
//!
//! Maps cleanly from an OpenClaw `SKILL.md` (YAML frontmatter `name`/`description`
//! + markdown body): the body becomes the skill's `procedures.md` (the natural
//! language source of truth). Deterministic — it persists the supplied content
//! and loads the skill; no internal LLM pass (LLM-assisted extraction is wired
//! separately and currently falls back to the stub regardless).

use std::collections::HashSet;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::skills::author::{ExtractionHints, SkillAuthor};

/// Author and install a skill from structured metadata + a markdown body.
pub struct SkillAuthorTool;

fn domains_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "description": "Domain tags for the skill (e.g. [\"kubernetes\", \"ops\"]).",
        "items": {"type": "string"}
    })
}

impl Tool for SkillAuthorTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "skill_author".into(),
            description: "Register a skill from an id, name, description, and a markdown body \
                          (e.g. an OpenClaw SKILL.md). The body is stored as the skill's \
                          procedures.md."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "id".into(),
                    description: "Stable skill id (kebab-case, e.g. \"cluster-inspect\").".into(),
                    required: true,
                },
                ToolParam {
                    name: "name".into(),
                    description: "Human-readable skill name.".into(),
                    required: true,
                },
                ToolParam {
                    name: "description".into(),
                    description: "One-line description of what the skill does / when to use it."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "procedures_md".into(),
                    description: "The full markdown body (the SKILL.md content below frontmatter)."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "domains".into(),
                    description: "Array of domain tags.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let id = input.require("id", "skill_author")?.to_string();
        let name = input.require("name", "skill_author")?.to_string();
        let description = input.require("description", "skill_author")?.to_string();
        let procedures_md = input.require("procedures_md", "skill_author")?.to_string();
        let domains = input.get_str_list("domains");

        // Persistence is required to write the skill directory.
        let data_dir =
            engine
                .config()
                .data_dir
                .clone()
                .ok_or_else(|| AgentError::ToolExecution {
                    tool_name: "skill_author".into(),
                    message: "skill authoring requires a persistent workspace (--data-dir)".into(),
                })?;
        let skills_dir = data_dir.join("skills");

        // Build a stub skill carrying the metadata, then graft the supplied body
        // in as the procedures.md source of truth.
        let author = SkillAuthor::new(false);
        let mut authored = author
            .author_from_description(
                &id,
                &name,
                &description,
                ExtractionHints {
                    domains,
                    ..Default::default()
                },
            )
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "skill_author".into(),
                message: e.to_string(),
            })?;
        authored.source_text = procedures_md;

        // Write all skill files (incl. procedures.md), then load via the engine.
        authored
            .write_to_disk(&skills_dir)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "skill_author".into(),
                message: e.to_string(),
            })?;
        let activation = engine
            .load_skill(&id)
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "skill_author".into(),
                message: e.to_string(),
            })?;

        Ok(ToolOutput::ok(format!(
            "Authored and loaded skill \"{id}\" ({} triple(s) loaded).",
            activation.triples_loaded
        )))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "skill_author".into(),
            description:
                "Register a skill from id/name/description/markdown body (e.g. a SKILL.md).".into(),
            parameters: vec![
                ToolParamSchema::required("id", "Stable skill id (kebab-case)."),
                ToolParamSchema::required("name", "Human-readable skill name."),
                ToolParamSchema::required("description", "One-line description / when to use."),
                ToolParamSchema::required(
                    "procedures_md",
                    "The full markdown body (SKILL.md content below frontmatter).",
                ),
                ToolParamSchema::optional_schema("domains", "Domain tags.", domains_schema()),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg, Capability::WriteFilesystem]),
                description: "Writes a skill directory and loads it into the engine.".into(),
                shadow_triggers: vec!["author skill".into(), "install skill".into()],
            },
            source: ToolSource::Native,
        }
    }
}
